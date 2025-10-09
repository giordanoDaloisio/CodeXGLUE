#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Valutazione pass@1, pass@10, pass@20 sul dataset HumanEval usando un modello fine-tunato.

Richiede il pacchetto `human-eval` (installare con: `pip install human-eval`).
Genera fino a `beam_size` completamenti per task usando la beam search già
implementata nel modello `Seq2Seq` (file `model.py`) e calcola le metriche pass@k.

Uso tipico:

python evaluate_passk.py \
  --model_dir ./finetuned_model \
  --data_file ../data/HumanEval.jsonl.gz \
  --beam_size 20 \
  --max_source_length 512 \
  --max_target_length 128 \
  --timeout 3.0 \
  --output_dir ./eval_passk

Output:
- Stampa metriche pass@1, pass@10, pass@20
- Salva `samples.jsonl` con tutte le completion generate
- Salva `metrics.json` con le metriche calcolate

Nota: La beam search produce completamenti deterministici (i 20 migliori secondo il modello).
Per una stima più realistica di pass@k si userebbero campionamenti stocastici (temperature / nucleus).
Qui seguiamo la richiesta esplicita usando il modello addestrato e beam search.
"""
from __future__ import annotations
import argparse
import json
import os
import tempfile
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Importiamo funzioni/strutture dal file di fine-tuning
from finetune_humaneval import (
    load_trained_model,
    read_humaneval,
    Example,
    HumanEvalDataset,
    Collator,
)


def build_samples(model, tokenizer, dataset: HumanEvalDataset, device: torch.device, beam_size: int) -> List[Dict[str, Any]]:
    """Genera fino a `beam_size` completamenti per ogni task e ritorna lista di dizionari
    {"task_id":..., "completion":...} compatibili con l'harness HumanEval.
    """
    dl = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=Collator(tokenizer.pad_token_id))
    model.eval()
    samples = []
    times = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="Generate-Beams"):
            source_ids = batch['source_ids'].to(device)
            source_mask = batch['source_mask'].to(device)
            start_time = time.time()
            beam_outputs = model(source_ids=source_ids, source_mask=source_mask)
            end_time = time.time()
            times.append(end_time - start_time)
            task_id = batch['task_ids'][0]
            # beam_outputs shape: (1, B, max_len)
            num_beams = min(beam_size, beam_outputs.shape[1])
            for b in range(num_beams):
                seq = beam_outputs[0, b].tolist()
                # Tronca a eos se presente
                if tokenizer.eos_token_id in seq:
                    seq = seq[:seq.index(tokenizer.eos_token_id)]
                completion = tokenizer.decode(seq, skip_special_tokens=True)
                samples.append({"task_id": task_id, "completion": completion})
    return samples, times


def evaluate_passk(model_dir: str, data_file: str, beam_size: int, max_source_length: int, max_target_length: int, timeout: float, output_dir: str):
    # Validazione parametri
    k_list = [1, 10, 20]
    if beam_size < max(k_list):
        raise ValueError(f"Per calcolare pass@{max(k_list)} serve beam_size >= {max(k_list)} (fornito {beam_size}).")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_trained_model(model_dir, beam_size=beam_size, max_target_length=max_target_length, device=device)

    raw = read_humaneval(data_file)
    if not raw:
        raise ValueError("File HumanEval vuoto o formattato male.")
    examples = [Example(task_id=r.get('task_id', f"task_{i}"), source=r['prompt'], target=r['canonical_solution']) for i,r in enumerate(raw)]
    dataset = HumanEvalDataset(examples, tokenizer, max_source_length, max_target_length)

    # Generiamo completamenti (samples)
    print("[Info] Generazione completamenti (beam search)...")
    samples, times = build_samples(model, tokenizer, dataset, device, beam_size)

    os.makedirs(output_dir, exist_ok=True)
    samples_file = os.path.join(output_dir, 'samples.jsonl')
    with open(samples_file, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    print(f"[Info] Salvato file samples: {samples_file}")

    with open(os.path.join(output_dir, 'generation_times.txt'), 'w', encoding='utf-8') as f:
        for t in times:
            f.write(f"{t}\n")
    print(f"[Info] Tempi di generazione medi per task: {sum(times)/len(times):.4f} secondi")
    
    # Valutazione funzionale
    try:
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError:
        print("[ERRORE] Il pacchetto 'human-eval' non è installato. Installarlo con: pip install human-eval")
        return

    print("[Info] Eseguo harness HumanEval (può richiedere alcuni minuti)...")
    metrics = evaluate_functional_correctness(
        samples_jsonl=samples_file,
        problem_file=None,
        k=k_list,
        timeout=timeout
    )

    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"[Risultati] pass@1={metrics.get('pass@1'):.4f} | pass@10={metrics.get('pass@10'):.4f} | pass@20={metrics.get('pass@20'):.4f}")
    print(f"[Info] Metriche salvate in: {metrics_file}")


def parse_args():
    ap = argparse.ArgumentParser(description="Calcolo pass@1/10/20 su HumanEval con modello fine-tunato")
    ap.add_argument('--model_dir', type=str, required=True, help='Directory con tokenizer e checkpoint (output fine-tuning)')
    ap.add_argument('--data_file', type=str, required=True, help='File HumanEval jsonl o jsonl.gz')
    ap.add_argument('--beam_size', type=int, default=20, help='Numero di beam/completamenti per task (>=20)')
    ap.add_argument('--max_source_length', type=int, default=512)
    ap.add_argument('--max_target_length', type=int, default=128)
    ap.add_argument('--timeout', type=float, default=3.0, help='Timeout per test funzione (secondi)')
    ap.add_argument('--output_dir', type=str, default='./eval_passk')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_passk(
        model_dir=args.model_dir,
        data_file=args.data_file,
        beam_size=args.beam_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        timeout=args.timeout,
        output_dir=args.output_dir
    )
