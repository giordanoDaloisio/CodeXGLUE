from human_eval.data import read_problems, write_jsonl
from transformers import QuantoConfig, AutoModelForCausalLM, AutoTokenizer
from hf_token import hf_token
from huggingface_hub import login
from time import time
import torch
import csv
import os
from argparse import ArgumentParser
from typing import List, Dict, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

login(hf_token)

def print_model_size(model):
    """Calcola la dimensione del modello senza salvare file temporanei"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1e6
    logger.info(f"Size (MB): {size_mb:.2f}")
    return size_mb

def generate_batch_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    task_ids: List[str],
    gen_kwargs: Dict,
    batch_size: int,
    model_device: Optional[torch.device] = None,
) -> tuple:
    """Genera completions in batch usando model.generate.

    Ritorna (samples, total_time_sec), dove samples è una lista di dict
    {task_id, completion}.
    """
    start_time = time()
    samples: List[Dict] = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_task_ids = task_ids[i : i + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Sposta gli input sul device del modello solo nel caso single-device
        if model_device is not None:
            enc = {k: v.to(model_device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                **gen_kwargs,
            )

        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for tid, text in zip(batch_task_ids, texts):
            samples.append({"task_id": tid, "completion": text})

    total_time = time() - start_time
    return samples, total_time

def write_samples_batch(samples: List[Dict], filename: str = "samples.jsonl"):
    """Scrive i samples in batch invece di uno alla volta"""
    with open(filename, "a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

def ensure_directory_exists(filepath: str):
    """Crea la directory se non esiste"""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--job_id", type=str, required=True, help="Job ID for tracking")
    parser.add_argument("--num_samples_per_task", type=int, default=100, help="Number of samples per task")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--quantf8", action="store_true", help="Use quantization for the model")
    parser.add_argument("--quant8", action="store_true", help="Use quantization for the model")
    parser.add_argument("--quant4", action="store_true", help="Use quantization for the model")

    args = parser.parse_args()
    job_id = args.job_id
    
    # Determina device_map in base alla quantizzazione
    if args.quantf8 or args.quant8 or args.quant4:
        device_map = None  # Carica tutto su un singolo dispositivo
        torch_dtype = torch.float16  # Usa float16 per risparmiare memoria
        quant_conf = QuantoConfig(weights="int8" if args.quant8 else "int4" if args.quant4 else "float8")

    else:
        device_map = "auto"
        torch_dtype = None
        quant_conf = None
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        quantization_config=quant_conf
    )

    model.eval()

    # Se non usiamo device_map (single device), mettiamo esplicitamente il modello su CUDA se disponibile
    model_device: Optional[torch.device] = None
    if device_map is None:
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(model_device)

    # pipe = pipeline(
    #     task="text-generation", 
    #     model="meta-llama/Llama-3.1-8B-Instruct", 
    #     device_map=device_map,
    #     torch_dtype=torch_dtype,
    #     trust_remote_code=True,
    #     config=quant_conf
    # )
    # pipe.tokenizer.pad_token = pipe.tokenizer.eos_token  # Imposta il token di padding
    times_file = f"times/times_{job_id}.csv"
    samples_file = f"samples_{job_id}.jsonl"
  
    # Leggi problemi normalmente
    problems = read_problems()
    
    # Stampa dimensione del modello
    model_size = print_model_size(model)
    logger.info(f"Model size: {model_size:.2f} MB")
    
    # Assicurati che le directory esistano
    ensure_directory_exists(times_file)
    
    # Prepara header del CSV se il file non esiste
    if not os.path.exists(times_file):
        with open(times_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_id", "batch_time", "samples_in_batch", "time_per_sample"])
    
    all_times = []
    all_samples = []
    
    # Prepara tutti i prompt in anticipo
    task_ids = list(problems.keys())

    ## GPU/Model Warmup
    logger.info("Eseguendo warmup del modello...")
    warmup_prompts = [problems[task_id]["prompt"] for task_id in task_ids[: min(10, len(task_ids))]]
    if warmup_prompts:
        enc_w = tokenizer(warmup_prompts, return_tensors="pt", padding=True, truncation=True)
        if model_device is not None:
            enc_w = {k: v.to(model_device) for k, v in enc_w.items()}
        with torch.no_grad():
            _ = model.generate(
                input_ids=enc_w["input_ids"],
                attention_mask=enc_w.get("attention_mask"),
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    logger.info("Warmup completato.")
    
    logger.info(f"Generando {args.num_samples_per_task} samples per {len(task_ids)} tasks...")
    
    # Parametri di generazione condivisi
    gen_kwargs = dict(
        do_sample=True,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Processa in batch per massimizzare l'efficienza
    batch_total_times: List[float] = []
    for sample_idx in range(args.num_samples_per_task):
        logger.info(f"Batch {sample_idx + 1}/{args.num_samples_per_task}")
        
        # Prepara batch di prompt (uno per ogni task)
        prompts = [problems[task_id]["prompt"] for task_id in task_ids]
        
        # Genera completions in batch
        batch_samples, batch_time = generate_batch_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            task_ids=task_ids,
            gen_kwargs=gen_kwargs,
            batch_size=max(1, int(args.batch_size)),
            model_device=model_device,
        )
        
        # Salva samples
        write_samples_batch(batch_samples, samples_file)
        all_samples.extend(batch_samples)
        
        # Calcola statistiche di timing
        time_per_sample = batch_time / len(batch_samples)
        all_times.append(time_per_sample)
        batch_total_times.append(batch_time)
        
        # Salva timing info
        with open(times_file, "a", newline="") as f:
            writer = csv.writer(f)
            for i, task_id in enumerate(task_ids):
                writer.writerow([task_id, batch_time, len(batch_samples), time_per_sample])
        
        logger.info(f"Batch time: {batch_time:.2f}s, Time per sample: {time_per_sample:.2f}s")
    
    # Statistiche finali
    total_samples = len(all_samples)
    avg_time = sum(all_times) / len(all_times) if all_times else 0.0
    total_time = sum(batch_total_times)
    
    logger.info(f"\nStatistiche finali:")
    logger.info(f"- Totale samples generati: {total_samples}")
    logger.info(f"- Tempo medio per sample: {avg_time:.2f}s")
    logger.info(f"- Tempo totale: {total_time:.2f}s")
    throughput = (total_samples / total_time) if total_time > 0 else 0.0
    logger.info(f"- Throughput: {throughput:.2f} samples/s")
    
    # Salva statistiche finali
    with open(f"stats_{job_id}.json", "w") as f:
        json.dump({
            "job_id": job_id,
            "total_samples": total_samples,
            "avg_time_per_sample": avg_time,
            "total_time": total_time,
            "throughput": throughput,
            "model_size_mb": model_size,
            "num_tasks": len(task_ids),
            "samples_per_task": args.num_samples_per_task
        }, f, indent=2)

