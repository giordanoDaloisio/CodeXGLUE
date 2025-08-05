from human_eval.data import read_problems, write_jsonl
from transformers import pipeline
from hf_token import hf_token
from huggingface_hub import login
from time import time
import torch
import csv
import os
from argparse import ArgumentParser
from typing import List, Dict
import json

login(hf_token)

def print_model_size(model):
    """Calcola la dimensione del modello senza salvare file temporanei"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1e6
    print(f"Size (MB): {size_mb:.2f}")
    return size_mb

def generate_batch_completions(pipe, prompts: List[str], task_ids: List[str]) -> tuple:
    """Genera completions in batch per maggiore efficienza"""
    start_time = time()
    # Usa batching per migliorare l'efficienza
    responses = pipe(
        prompts,
        do_sample=True,
        pad_token_id=pipe.tokenizer.eos_token_id,
        batch_size=min(8, len(prompts))  # Batch size adattivo
    )
    
    end_time = time()
    total_time = end_time - start_time
    
    # Prepara i samples
    samples = []
    for task_id, response in zip(task_ids, responses):
        completion = response[0]["generated_text"] if isinstance(response, list) else response["generated_text"]        
        samples.append({
            "task_id": task_id,
            "completion": completion
        })
    
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
    
    args = parser.parse_args()
    job_id = args.job_id
    
    # Inizializza la pipeline con ottimizzazioni
    pipe = pipeline(
        task="text-generation", 
        model="meta-llama/Llama-3.1-8B-Instruct", 
        device_map="auto",
        trust_remote_code=True
    )
    
    # Stampa dimensione del modello
    model_size = print_model_size(pipe.model)
    print(f"Model size: {model_size:.2f} MB")
    
    # Leggi problemi
    problems = read_problems()
    
    # Assicurati che le directory esistano
    times_file = f"times/times_{job_id}.csv"
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

    ## GPU Warmup
    print("Eseguendo warmup della GPU...")
    warmup_prompts = [problems[task_id]["prompt"] for task_id in task_ids[:10]]
    pipe(warmup_prompts, do_sample=True)
    print("Warmup completato.")
    
    print(f"Generando {args.num_samples_per_task} samples per {len(task_ids)} tasks...")
    
    # Processa in batch per massimizzare l'efficienza
    for sample_idx in range(args.num_samples_per_task):
        print(f"Batch {sample_idx + 1}/{args.num_samples_per_task}")
        
        # Prepara batch di prompt (uno per ogni task)
        prompts = [problems[task_id]["prompt"] for task_id in task_ids]
        
        # Genera completions in batch
        batch_samples, batch_time = generate_batch_completions(
            pipe, prompts, task_ids, args.max_new_tokens
        )
        
        # Salva samples
        write_samples_batch(batch_samples)
        all_samples.extend(batch_samples)
        
        # Calcola statistiche di timing
        time_per_sample = batch_time / len(batch_samples)
        all_times.append(time_per_sample)
        
        # Salva timing info
        with open(times_file, "a", newline="") as f:
            writer = csv.writer(f)
            for i, task_id in enumerate(task_ids):
                writer.writerow([task_id, batch_time, len(batch_samples), time_per_sample])
        
        print(f"Batch time: {batch_time:.2f}s, Time per sample: {time_per_sample:.2f}s")
    
    # Statistiche finali
    total_samples = len(all_samples)
    avg_time = sum(all_times) / len(all_times)
    total_time = sum(all_times) * len(task_ids)
    
    print(f"\nStatistiche finali:")
    print(f"- Totale samples generati: {total_samples}")
    print(f"- Tempo medio per sample: {avg_time:.2f}s")
    print(f"- Tempo totale: {total_time:.2f}s")
    print(f"- Throughput: {total_samples/total_time:.2f} samples/s")
    
    # Salva statistiche finali
    with open(f"stats_{job_id}.json", "w") as f:
        json.dump({
            "job_id": job_id,
            "total_samples": total_samples,
            "avg_time_per_sample": avg_time,
            "total_time": total_time,
            "throughput": total_samples/total_time,
            "model_size_mb": model_size,
            "num_tasks": len(task_ids),
            "samples_per_task": args.num_samples_per_task
        }, f, indent=2)

