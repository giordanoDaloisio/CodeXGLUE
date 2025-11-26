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
from torch.nn.utils import prune

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

login(hf_token)

def print_model_size(model):
    torch.save(model.state_dict(), "tmp.p")
    print("Size (MB): " + str(os.path.getsize("tmp.p") / 1e6))
    os.remove("tmp.p")

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
    times = []
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
            start_time = time()
            outputs = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                **gen_kwargs,
            )

        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for tid, text in zip(batch_task_ids, texts):
            samples.append({"task_id": tid, "completion": text})

        batch_time = time() - start_time
        times.append(batch_time)

    return samples, times

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
    parser.add_argument("--prune6", action="store_true", help="Use prune for the model")
    parser.add_argument("--prune4", action="store_true", help="Use prune for the model")
    parser.add_argument("--prune", action="store_true", help="Use prune for the model")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")

    args = parser.parse_args()
    job_id = args.job_id
    
    # Determina device_map in base alla quantizzazione
    if args.quantf8 or args.quant8 or args.quant4:
        device_map = None  # Carica tutto su un singolo dispositivo
        torch_dtype = torch.float16  # Usa float16 per risparmiare memoria
        quant_conf = QuantoConfig(weights="int8" if args.quant8 else "int4" if args.quant4 else "float8")

    else:
        device_map = None
        torch_dtype = None
        quant_conf = None
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        quantization_config=quant_conf
    )

    if args.prune6:
            logger.info("******* Apply Pruning 0.6 ***********")
            parameters_to_prune = []
            
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, "weight"))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.6,
            )
            for module, param in parameters_to_prune:
                prune.remove(module, param)

    if args.prune4:
        logger.info("******* Apply Pruning 0.4 ***********")
        parameters_to_prune = []
        
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.4,
        )
        for module, param in parameters_to_prune:
            logger.info(prune.is_pruned(module))
            prune.remove(module, param)
            logger.info(prune.is_pruned(module))

    if args.prune:
        logger.info("******* Apply Pruning 0.2 ***********")
        parameters_to_prune = []
        
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.2,
        )
        for module, param in parameters_to_prune:
            prune.remove(module, param)
 

    model.eval()

    # Se non usiamo device_map (single device), mettiamo esplicitamente il modello su CUDA se disponibile
    model_device: Optional[torch.device] = None
    if device_map is None:
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(model_device)

    times_file = f"times/times_{job_id}.csv"
    samples_file = f"samples_{job_id}.jsonl"
  
    # Leggi problemi normalmente
    problems = read_problems()
    
    # Stampa dimensione del modello
    print_model_size(model)
    
    # Assicurati che le directory esistano
    ensure_directory_exists(times_file)
    
    # # Prepara header del CSV se il file non esiste
    if not os.path.exists(times_file):
        with open(times_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_id", "batch_time", "samples_in_batch"])
    
    all_times = []
    all_samples = []
    
    # # Prepara tutti i prompt in anticipo
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
    
    # # Parametri di generazione condivisi
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
        batch_samples, batch_times = generate_batch_completions(
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
        # time_per_sample = batch_time / len(batch_samples)
        # all_times.append(time_per_sample)
        batch_total_times.extend(batch_times)
        
        # Salva timing info
        with open(times_file, "a", newline="") as f:
            writer = csv.writer(f)
            num_tasks = len(task_ids)
            for i, batch_time in enumerate(batch_times):
                start = i * args.batch_size
                end = min(start + args.batch_size, num_tasks)
                bs = max(0, end - start)
                writer.writerow([f"batch_{sample_idx}_{i}", batch_time, bs])
        
        # logger.info(f"Batch time: {batch_time:.2f}s, Time per sample: {time_per_sample:.2f}s")
    
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
            "num_tasks": len(task_ids),
            "samples_per_task": args.num_samples_per_task
        }, f, indent=2)

