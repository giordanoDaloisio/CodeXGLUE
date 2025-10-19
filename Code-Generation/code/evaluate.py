#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script standalone per valutare un modello CodeGPT (o decoder-only) su HumanEval.
Può essere usato su modelli già fine-tunati o modelli base.

Usage:
python evaluate_model_humaneval.py \
  --model_path ./codegpt_finetuned \
  --humaneval_file ../data/HumanEval.jsonl.gz \
  --output_dir ./eval_results \
  --num_samples_per_task 20 \
  --temperature 0.8
"""
import argparse
import json
import os
import gzip
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from tqdm import tqdm
import logging
import time
from optimum.quanto import qint8, qint4, qfloat8, quantize, freeze, Calibration


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_model_size(model):
    """Calcola la dimensione del modello senza salvare file temporanei"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1e6
    logger.info(f"Size (MB): {size_mb:.2f}")
    return size_mb

def load_humaneval(path: str) -> List[Dict[str, Any]]:
    """Carica dataset HumanEval."""
    opener = gzip.open if path.endswith('.gz') else open
    problems = []
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems

def calibration(problems, tokenizer, model, device, args):
       logger.info("Calibrating the model...")
       with torch.no_grad():
        for i, problem in enumerate(problems):
            if i < 10:
                prompt = problem['prompt']
                
                # Tokenizza prompt
                inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                # Genera multiple samples
                _ = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens= args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    num_return_sequences=args.num_samples_per_task,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )


def generate_completions(
    model,
    tokenizer,
    problems: List[Dict[str, Any]],
    num_samples_per_task: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str
) -> List[Dict[str, str]]:
    """Genera completions per tutti i problemi."""
    samples = []
    times = []
    model.eval()
    
    with torch.no_grad():
        for problem in tqdm(problems, desc="Generating"):
            prompt = problem['prompt']
            task_id = problem['task_id']
            
            # Tokenizza prompt
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            prompt_len = input_ids.shape[1]
            start_time = time.time()
            # Genera multiple samples
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_samples_per_task,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            end_time = time.time()
            times.append(end_time - start_time)
            # Decodifica completions (rimuovi prompt)
            for output in outputs:
                completion_ids = output[prompt_len:]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                
                # Post-processing: stop at next function definition
                lines = completion.split('\n')
                cleaned_lines = []
                for line in lines:
                    # Stop se troviamo una nuova funzione/classe e abbiamo già del codice
                    if line.strip().startswith(('def ', 'class ', 'import ', 'from ')) and cleaned_lines:
                        break
                    cleaned_lines.append(line)
                
                completion = '\n'.join(cleaned_lines)
                
                samples.append({
                    'task_id': task_id,
                    'completion': completion
                })
    
    return samples, times


def evaluate_humaneval(
    model_path: str,
    humaneval_file: str,
    output_dir: str,
    num_samples_per_task: int = 20,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    timeout: float = 3.0,
    use_fp16: bool = True,
    args = None
):
    """Valuta modello su HumanEval e calcola pass@k."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Info] Device: {device}")

    if args.quantize_f8 or args.quantize_i8 or args.quantize_i4:
        device_map = None  # Carica tutto su un singolo dispositivo
        quant_conf = QuantoConfig(weights="int8" if args.quantize_i8 else "int4" if args.quantize_i4 else "float8")
    else:
        device_map = None
        quant_conf = None
    
    # Carica modello
    print(f"[Info] Carico modello da {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Gestisci pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Carica modello con dtype appropriato
    dtype = torch.float16 if use_fp16 and device == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quant_conf
    ).to(device)
        
    # Carica HumanEval
    print(f"[Info] Carico HumanEval da {humaneval_file}")
    problems = load_humaneval(humaneval_file)
    print(f"[Info] Num problemi: {len(problems)}")


    # if args.quantize_i8:
    #     logger.info("********** Apply Quantization qint8 **********")
    #     quantize(model, weights=qint8, activations=qint8)
    #     with Calibration():
    #         logger.info("*********** Calibrate **************")
    #         calibration(problems, tokenizer, model, device, args)
    #     freeze(model)

    # if args.quantize_i4:
    #     logger.info("********** Apply Quantization qint4 **********")
    #     quantize(model, weights=qint4, activations=qint4)
    #     with Calibration():
    #         logger.info("*********** Calibrate **************")
    #         calibration(problems, tokenizer, model, device, args)
    #     freeze(model)

    # if args.quantize_f8:
    #     logger.info("********** Apply Quantization qfloat8 **********")
    #     quantize(model, weights=qfloat8, activations=qfloat8)
    #     with Calibration():
    #         logger.info("*********** Calibrate **************")
    #         calibration(problems, tokenizer, model, device, args)
    #     freeze(model)
    

    print(f"[Info] Modello caricato: {model.config.model_type}")
    print(f"[Info] Num parameters: {model.num_parameters():,}")
    print((f"[Info] Dimensione del modello: {print_model_size(model):,}"))

    # Genera completions
    print(f"[Info] Generazione completions (samples per task: {num_samples_per_task})")
    samples, times = generate_completions(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        num_samples_per_task=num_samples_per_task,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device
    )
    
    # Salva samples
    os.makedirs(output_dir, exist_ok=True)
    samples_file = os.path.join(output_dir, 'samples.jsonl')
    with open(samples_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    # Salva tempi di generazione
    times_file = os.path.join(output_dir, 'times.jsonl')
    if args.quantize_i8:
         times_file = os.path.join(output_dir, 'times_qint8.jsonl')
    if args.quantize_i4:
            times_file = os.path.join(output_dir, 'times_qint4.jsonl')
    if args.quantize_f8:
            times_file = os.path.join(output_dir, 'times_qfloat8.jsonl')
    with open(times_file, 'w', encoding='utf-8') as f:
        for t in times:
            f.write(json.dumps(t) + '\n')

    print(f"[Info] Salvati {len(samples)} samples in {samples_file}")
    print(f"[Info] Salvati {len(times)} tempi di generazione in {times_file}")

    # Salva anche configurazione della valutazione
    config_file = os.path.join(output_dir, 'eval_config.json')
    eval_config = {
        'model_path': model_path,
        'num_samples_per_task': num_samples_per_task,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'num_problems': len(problems),
        'total_samples': len(samples),
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(eval_config, f, indent=2)
    
    # Valutazione funzionale con human_eval
    try:
        from human_eval.evaluation import evaluate_functional_correctness
        
        print(f"[Info] Eseguo valutazione funzionale (timeout={timeout}s)")
        k_values = [1, 10, 20] if num_samples_per_task >= 20 else [1, 10]
        
        results = evaluate_functional_correctness(
            sample_file=samples_file,
            k=k_values,
            timeout=timeout,
        )
        
        # Salva risultati
        results_file = os.path.join(output_dir, 'results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Stampa risultati
        print(f"\n{'='*70}")
        print("HUMANEVAL EVALUATION RESULTS")
        print(f"{'='*70}")
        for metric, value in results.items():
            print(f"  {metric:20s}: {value:.4f} ({value*100:.2f}%)")
        print(f"{'='*70}\n")
        
        print(f"[Info] Risultati salvati in {results_file}")
        return results
        
    except ImportError:
        print("\n[WARNING] Pacchetto 'human-eval' non installato!")
        print("Installare con: pip install human-eval")
        print(f"\nSamples salvati in {samples_file}")
        print("Eseguire manualmente la valutazione con:")
        print(f"  evaluate_functional_correctness {samples_file}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Valuta modello decoder-only su HumanEval")
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path al modello fine-tunato (o nome HF)')
    parser.add_argument('--humaneval_file', type=str, required=True,
                       help='Path a HumanEval.jsonl.gz')
    parser.add_argument('--output_dir', type=str, default='./humaneval_eval',
                       help='Directory output per samples e risultati')
    
    parser.add_argument('--num_samples_per_task', type=int, default=20,
                       help='Numero di samples da generare per task (min 20 per pass@20)')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help='Max token da generare per completion')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Temperature per sampling (0.6-0.9 raccomandato)')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Nucleus sampling threshold')
    parser.add_argument('--timeout', type=float, default=3.0,
                       help='Timeout per test execution (secondi)')
    parser.add_argument('--no_fp16', action='store_true',
                       help='Disabilita FP16 (usa FP32)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device da usare (default: cuda se disponibile)')
    parser.add_argument('--quantize_f8', action='store_true')
    parser.add_argument('--quantize_i8', action='store_true')
    parser.add_argument('--quantize_i4', action='store_true')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    evaluate_humaneval(
        model_path=args.model_path,
        humaneval_file=args.humaneval_file,
        output_dir=args.output_dir,
        num_samples_per_task=args.num_samples_per_task,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        use_fp16=not args.no_fp16,
        args = args
    )
