#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Distillation di CodeGPT su DistilGPT (o altro modello decoder-only) su CodeSearchNet-Python
per code generation, seguito da valutazione su HumanEval.

Workflow:
1. Carica dataset CodeSearchNet-Python (docstring + code)
2. Knowledge Distillation con causal LM training
3. Salva checkpoint
4. Valuta su HumanEval con pass@k metrics

Usage:
python finetune_codegpt_codesearchnet.py \
  --model_name_or_path microsoft/CodeGPT-small-py \
  --output_dir ./codegpt_finetuned \
  --num_train_epochs 3 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --learning_rate 5e-5 \
  --max_seq_length 512 \
  --do_train \
  --do_eval \
  --eval_on_humaneval
"""
from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from tqdm import tqdm


# ==========================================================
# Dataset Preparation
# ==========================================================

class CodeSearchNetDataset(Dataset):
    """Dataset per CodeSearchNet-Python in formato causal LM."""
    
    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512,
        mode: str = 'train'
    ):
        """
        Args:
            examples: Lista di dict con 'func_documentation_string' e 'func_code_string'
            tokenizer: Tokenizer del modello
            max_length: Lunghezza massima sequenza
            mode: 'train' o 'eval'
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Formato: docstring + code completo (causal LM)
        # Il modello impara a generare tutto dato un prefisso
        docstring = ex.get('func_documentation_string', '')
        code = ex.get('func_code_string', '')
        
        # Formatta come: """docstring"""\ncode
        if docstring:
            text = f'"""{docstring}"""\n{code}'
        else:
            text = code
        
        # Tokenizza
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        # Per causal LM: input_ids = labels
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
        }


def load_codesearchnet_python(
    split: str = 'train',
    num_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Carica CodeSearchNet-Python da HuggingFace datasets.
    
    Args:
        split: 'train', 'validation', o 'test'
        num_samples: Se specificato, limita a N esempi (per debug)
        cache_dir: Directory cache per dataset
    
    Returns:
        Lista di dizionari con func_documentation_string e func_code_string
    """
    print(f"[Info] Carico CodeSearchNet-Python split={split}")
    
    # Carica da HF datasets
    dataset = load_dataset(
        'code_search_net',
        'python',
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Filtra esempi validi (con docstring e code non vuoti)
    examples = []
    for item in tqdm(dataset, desc="Preprocessing"):
        doc = item.get('func_documentation_string', '').strip()
        code = item.get('func_code_string', '').strip()
        
        # Filtra esempi troppo corti o vuoti
        if doc and code and len(code) > 20:
            examples.append({
                'func_documentation_string': doc,
                'func_code_string': code,
                'func_name': item.get('func_name', ''),
            })
        
        if num_samples and len(examples) >= num_samples:
            break
    
    print(f"[Info] Caricati {len(examples)} esempi validi")
    return examples


# ==========================================================
# Training
# ==========================================================

def train_model(args):
    """Fine-tune CodeGPT su CodeSearchNet-Python."""
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    print(f"[Info] Device: {device}")
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Carica tokenizer e modello
    print(f"[Info] Carico modello da {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # GPT-2 non ha pad_token di default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    
    # Resize embeddings se il tokenizer è stato modificato
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"[Info] Modello: {config.model_type}, Params: {model.num_parameters():,}")
    
    # Carica dataset
    print(f"[Info] Carico dataset CodeSearchNet-Python")
    train_examples = load_codesearchnet_python(
        split='train',
        num_samples=args.max_train_samples,
        cache_dir=args.cache_dir
    )
    
    eval_examples = load_codesearchnet_python(
        split='validation',
        num_samples=args.max_eval_samples,
        cache_dir=args.cache_dir
    )
    
    # Crea datasets
    train_dataset = CodeSearchNetDataset(
        train_examples,
        tokenizer,
        max_length=args.max_seq_length,
        mode='train'
    )
    
    eval_dataset = CodeSearchNetDataset(
        eval_examples,
        tokenizer,
        max_length=args.max_seq_length,
        mode='eval'
    )
    
    print(f"[Info] Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Data collator per causal LM (gestisce padding dinamico)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, non masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy='steps' if args.do_eval else 'no',
        eval_steps=args.eval_steps if args.do_eval else None,
        save_strategy='steps',
        load_best_model_at_end=args.do_eval,
        metric_for_best_model='eval_loss' if args.do_eval else None,
        greater_is_better=False,
        fp16=args.fp16 and device == 'cuda',
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        report_to=args.report_to,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        data_collator=data_collator,
    )
    
    # Training
    if args.do_train:
        print(f"\n[Training] Inizio fine-tuning...")
        train_result = trainer.train()
        
        # Salva modello finale
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Salva metriche
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        print(f"[Training] Completato. Modello salvato in {args.output_dir}")
    
    # Evaluation su dataset CodeSearchNet
    if args.do_eval:
        print(f"\n[Evaluation] Valutazione su CodeSearchNet validation set...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print(f"[Evaluation] Eval loss: {metrics.get('eval_loss', 'N/A'):.4f}")
    
    # Evaluation su HumanEval
    if args.eval_on_humaneval:
        print(f"\n[HumanEval] Valutazione su HumanEval benchmark...")
        evaluate_on_humaneval(
            model_path=args.output_dir,
            humaneval_file=args.humaneval_file,
            output_dir=os.path.join(args.output_dir, 'humaneval_results'),
            num_samples_per_task=args.humaneval_num_samples,
            max_new_tokens=args.humaneval_max_tokens,
            temperature=args.humaneval_temperature,
            device=device
        )
    
    print("\n[Done] Pipeline completato!")


# ==========================================================
# HumanEval Evaluation
# ==========================================================

def evaluate_on_humaneval(
    model_path: str,
    humaneval_file: str,
    output_dir: str,
    num_samples_per_task: int = 20,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    device: str = 'cuda'
):
    """Valuta il modello fine-tunato su HumanEval."""
    import gzip
    
    # Carica modello
    print(f"[HumanEval] Carico modello da {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    ).to(device)
    model.eval()
    
    # Carica HumanEval
    opener = gzip.open if humaneval_file.endswith('.gz') else open
    problems = []
    with opener(humaneval_file, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    
    print(f"[HumanEval] Caricati {len(problems)} problemi")
    
    # Genera completions
    samples = []
    with torch.no_grad():
        for problem in tqdm(problems, desc="Generating"):
            prompt = problem['prompt']
            task_id = problem['task_id']
            
            # Tokenizza
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
            input_ids = inputs['input_ids'].to(device)
            prompt_len = input_ids.shape[1]
            
            # Genera multiple samples
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=num_samples_per_task,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decodifica
            for output in outputs:
                completion = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                
                # Post-processing: stop at next function
                lines = completion.split('\n')
                cleaned = []
                for line in lines:
                    if line.strip().startswith(('def ', 'class ')) and cleaned:
                        break
                    cleaned.append(line)
                completion = '\n'.join(cleaned)
                
                samples.append({
                    'task_id': task_id,
                    'completion': completion
                })
    
    # Salva samples
    os.makedirs(output_dir, exist_ok=True)
    samples_file = os.path.join(output_dir, 'samples.jsonl')
    with open(samples_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"[HumanEval] Salvati {len(samples)} samples in {samples_file}")
    
    # Valutazione con human_eval (se disponibile)
    try:
        from human_eval.evaluation import evaluate_functional_correctness
        
        print(f"[HumanEval] Eseguo valutazione funzionale...")
        k_list = [1, 10, 20] if num_samples_per_task >= 20 else [1, 10]
        results = evaluate_functional_correctness(
            sample_file=samples_file,
            k=k_list,
            timeout=3.0
        )
        
        results_file = os.path.join(output_dir, 'results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("HUMANEVAL RESULTS:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        print(f"{'='*60}\n")
        
        return results
        
    except ImportError:
        print("[Warn] Pacchetto 'human-eval' non installato.")
        print("       Samples salvati, eseguire manualmente:")
        print(f"       pip install human-eval")
        print(f"       evaluate_functional_correctness {samples_file}")
        return None


# ==========================================================
# Main & Args
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CodeGPT su CodeSearchNet-Python")
    
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/CodeGPT-small-py',
                       help='Pretrained model name or path')
    parser.add_argument('--output_dir', type=str, default='./codegpt_finetuned',
                       help='Output directory per modello e checkpoints')
    
    # Dataset
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory per datasets')
    parser.add_argument('--max_train_samples', type=int, default=None,
                       help='Max training samples (None = tutti)')
    parser.add_argument('--max_eval_samples', type=int, default=None,
                       help='Max eval samples (None = tutti)')
    parser.add_argument('--max_seq_length', type=int, default=512,
                       help='Max sequence length')
    
    # Training
    parser.add_argument('--do_train', action='store_true',
                       help='Esegui training')
    parser.add_argument('--do_eval', action='store_true',
                       help='Esegui evaluation su CodeSearchNet validation')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                       help='Numero epoch training')
    parser.add_argument('--train_batch_size', type=int, default=8,
                       help='Training batch size per device')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                       help='Eval batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    
    # Logging & Saving
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Log ogni N steps')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='Salva checkpoint ogni N steps')
    parser.add_argument('--eval_steps', type=int, default=1000,
                       help='Valuta ogni N steps')
    parser.add_argument('--save_total_limit', type=int, default=2,
                       help='Max numero checkpoints da tenere')
    parser.add_argument('--report_to', type=str, default='none',
                       help='Logging backend (tensorboard, wandb, none)')
    
    # Misc
    parser.add_argument('--fp16', action='store_true',
                       help='Usa mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Dataloader num workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Non usare CUDA anche se disponibile')
    
    # HumanEval
    parser.add_argument('--eval_on_humaneval', action='store_true',
                       help='Valuta su HumanEval dopo training')
    parser.add_argument('--humaneval_file', type=str, default='../data/HumanEval.jsonl.gz',
                       help='Path a HumanEval dataset')
    parser.add_argument('--humaneval_num_samples', type=int, default=20,
                       help='Num samples per task su HumanEval')
    parser.add_argument('--humaneval_max_tokens', type=int, default=256,
                       help='Max new tokens per completion')
    parser.add_argument('--humaneval_temperature', type=float, default=0.8,
                       help='Temperature per sampling su HumanEval')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(args)
