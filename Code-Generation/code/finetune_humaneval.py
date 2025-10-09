#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tuning del modello Seq2Seq definito in `model.py` sul dataset HumanEval.

Funzionalità principali:
- Caricamento del dataset HumanEval (jsonl o jsonl.gz) direttamente dal file scaricato da https://github.com/openai/human-eval
- Preparazione esempi (prompt -> soluzione canonica) per addestramento encoder-decoder
- Loop di training con logging, salvataggio checkpoint e early stopping opzionale
- Generazione con beam search (usa il metodo già implementato in `Seq2Seq`)
- Funzione di test che (se disponibile) usa `human_eval.evaluation.evaluate_functional_correctness` per calcolare pass@k
  altrimenti fallback su esatto match testuale (pass@1)

Esempio di esecuzione:

python finetune_humaneval.py \
  --data_file ../data/HumanEval.jsonl.gz \
  --pretrained_model microsoft/codebert-base \
  --output_dir ./finetuned_model \
  --num_train_epochs 3 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --beam_size 5

Nota: Il dataset HumanEval è piccolo (164 problemi). Il fine-tuning può facilmente overfittare: usare poche epoch o early stopping.
"""
from __future__ import annotations
import argparse
import gzip
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from transformers import AutoConfig, AutoTokenizer, RobertaModel
from tqdm import tqdm

# Import del modello Seq2Seq già fornito
from model import Seq2Seq

# ==========================================================
# Data Handling
# ==========================================================

def read_humaneval(path: str) -> List[Dict[str, Any]]:
    """Legge il file HumanEval (.jsonl o .jsonl.gz).
    Ogni riga contiene: task_id, prompt, canonical_solution, test, entry_point (a volte).
    Ritorna una lista di dizionari.
    """
    opener = gzip.open if path.endswith('.gz') else open
    data = []
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                js = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'prompt' in js and 'canonical_solution' in js:
                data.append(js)
    return data

@dataclass
class Example:
    task_id: str
    source: str
    target: str

class HumanEvalDataset(Dataset):
    def __init__(self, examples: List[Example], tokenizer, max_source_len: int, max_target_len: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Tokenizzazione sorgente e target (aggiungiamo i token speciali se necessari)
        src_ids = self.tokenizer.encode(ex.source, add_special_tokens=True, truncation=True, max_length=self.max_source_len)
        tgt_ids = self.tokenizer.encode(ex.target, add_special_tokens=True, truncation=True, max_length=self.max_target_len)
        return {
            'task_id': ex.task_id,
            'source_ids': torch.tensor(src_ids, dtype=torch.long),
            'target_ids': torch.tensor(tgt_ids, dtype=torch.long)
        }

class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, Any]]):
        source_ids = [b['source_ids'] for b in batch]
        target_ids = [b['target_ids'] for b in batch]
        source_ids = pad_sequence(source_ids, batch_first=True, padding_value=self.pad_id)
        target_ids = pad_sequence(target_ids, batch_first=True, padding_value=self.pad_id)
        # Mask (1 per token vero, 0 per pad)
        source_mask = (source_ids != self.pad_id).long()
        target_mask = (target_ids != self.pad_id).long()
        task_ids = [b['task_id'] for b in batch]
        return {
            'task_ids': task_ids,
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'target_mask': target_mask
        }

# ==========================================================
# Training & Evaluation Helpers
# ==========================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model_and_tokenizer(pretrained: str, beam_size: int, max_target_length: int, device: torch.device):
    """Crea tokenizer, encoder (Roberta/CodeBERT), decoder Transformer e wrapper Seq2Seq."""
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    if tokenizer.pad_token is None:
        # Per Roberta <pad> esiste già, ma se mancasse usiamo eos
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.sep_token

    config = AutoConfig.from_pretrained(pretrained)
    encoder = RobertaModel.from_pretrained(pretrained, config=config)

    # Costruzione decoder semplice (TransformerDecoder) compatibile con il forward definito in model.py
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    sos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id

    model = Seq2Seq(encoder=encoder,
                    decoder=decoder,
                    config=config,
                    beam_size=beam_size,
                    max_length=max_target_length,
                    sos_id=sos_id,
                    eos_id=eos_id)
    model.to(device)
    return model, tokenizer


def save_checkpoint(model: Seq2Seq, tokenizer, output_dir: str, step: Optional[int] = None):
    os.makedirs(output_dir, exist_ok=True)
    tag = f"step_{step}" if step is not None else "final"
    model_path = os.path.join(output_dir, f"model_{tag}.pt")
    torch.save({'model_state_dict': model.state_dict()}, model_path)
    tokenizer.save_pretrained(output_dir)
    # Salviamo anche la config HF dell'encoder se non già presente (serve per ricostruire architettura)
    config_file = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_file):
        try:
            model.config.save_pretrained(output_dir)
        except Exception as e:
            print(f"[Warn] Impossibile salvare config.json: {e}")
    print(f"[Checkpoint] Salvato: {model_path}")


def generate_predictions(model: Seq2Seq, tokenizer, dataloader: DataLoader, device: torch.device, max_len: int) -> Dict[str, str]:
    model.eval()
    preds = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generate"):
            source_ids = batch['source_ids'].to(device)
            source_mask = batch['source_mask'].to(device)
            beam_outputs = model(source_ids=source_ids, source_mask=source_mask)
            # beam_outputs shape: (batch, beam, max_length)
            for i, task_id in enumerate(batch['task_ids']):
                beam_seq = beam_outputs[i, 0].tolist()  # prendiamo il primo beam
                # Convertiamo in testo fermandoci a eos
                if tokenizer.eos_token_id in beam_seq:
                    beam_seq = beam_seq[:beam_seq.index(tokenizer.eos_token_id)]
                text = tokenizer.decode(beam_seq, skip_special_tokens=True)
                preds[task_id] = text
    return preds


def evaluate_functional(model: Seq2Seq, tokenizer, dataset: HumanEvalDataset, device: torch.device, beam_size: int, max_target_length: int, pass_ks=(1,10), timeout: float = 3.0):
    """Prova a valutare pass@k usando il pacchetto human_eval se presente. Ritorna dizionario metriche."""
    try:
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError:
        print("[Avviso] Pacchetto human-eval non installato. Fallback a match testuale semplice (pass@1). Installare con: pip install human-eval")
        dl = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=Collator(dataset.tokenizer.pad_token_id))
        preds = generate_predictions(model, tokenizer, dl, device, max_target_length)
        # Semplice pass@1: exact match canonical_solution
        gold_map = {ex.task_id: ex.target for ex in dataset.examples}
        correct = sum(1 for k,v in preds.items() if k in gold_map and v.strip() == gold_map[k].strip())
        return {"pass@1": correct / max(1,len(gold_map))}

    # Se pacchetto presente: generiamo più samples per ogni task (beam) e costruiamo file jsonl per evaluate_functional_correctness
    print("[Info] Valutazione funzionale HumanEval (potrebbe richiedere tempo) ...")
    model.eval()
    samples = []
    dl = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=Collator(dataset.tokenizer.pad_token_id))
    with torch.no_grad():
        for batch in tqdm(dl, desc="Gen-Eval"):
            source_ids = batch['source_ids'].to(device)
            source_mask = batch['source_mask'].to(device)
            beam_outputs = model(source_ids=source_ids, source_mask=source_mask)
            task_id = batch['task_ids'][0]
            # beam_outputs: (1, beam, max_len)
            for b in range(min(beam_size, beam_outputs.shape[1])):
                seq = beam_outputs[0, b].tolist()
                if tokenizer.eos_token_id in seq:
                    seq = seq[:seq.index(tokenizer.eos_token_id)]
                completion = tokenizer.decode(seq, skip_special_tokens=True)
                # Il completion deve essere l'integrazione finale del prompt -> official harness concatena prompt + completion
                samples.append({"task_id": task_id, "completion": completion})

    # Salviamo temporaneamente
    tmp_dir = os.path.join("/tmp", "humaneval_eval")
    os.makedirs(tmp_dir, exist_ok=True)
    samples_file = os.path.join(tmp_dir, "samples.jsonl")
    with open(samples_file, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"[Info] File samples creato: {samples_file}")

    results = evaluate_functional_correctness(
        samples_jsonl=samples_file,
        problem_file=None,  # Usa quello interno del pacchetto
        k=list(pass_ks),
        timeout=timeout
    )
    return results

# ==========================================================
# Main Training Loop
# ==========================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    set_seed(args.seed)

    print("[Info] Carico dataset...")
    raw = read_humaneval(args.data_file)
    if len(raw) == 0:
        raise ValueError("Dataset vuoto o file non valido")
    examples = [Example(task_id=r.get('task_id', f"task_{i}"), source=r['prompt'], target=r['canonical_solution']) for i,r in enumerate(raw)]

    # Split train/eval se non viene fornita percentuale disabilitata
    if args.eval_split > 0.0:
        split_idx = int(len(examples) * (1 - args.eval_split))
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]
    else:
        train_examples = examples
        eval_examples = examples

    print(f"[Info] Train size: {len(train_examples)} | Eval size: {len(eval_examples)}")

    model, tokenizer = build_model_and_tokenizer(
        pretrained=args.pretrained_model,
        beam_size=args.beam_size,
        max_target_length=args.max_target_length,
        device=device
    )

    # Salvataggio immediato di meta-informazioni per il caricamento successivo
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "meta.json")
    if not os.path.exists(meta_path):
        meta = {
            "pretrained_model": args.pretrained_model,
            "beam_size": args.beam_size,
            "max_target_length": args.max_target_length,
            "sos_id": model.sos_id,
            "eos_id": model.eos_id
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        # Salviamo anche la config per evitare errori di caricamento
        try:
            model.config.save_pretrained(args.output_dir)
        except Exception as e:
            print(f"[Warn] Config non salvata all'inizio: {e}")

    train_ds = HumanEvalDataset(train_examples, tokenizer, args.max_source_length, args.max_target_length)
    eval_ds = HumanEvalDataset(eval_examples, tokenizer, args.max_source_length, args.max_target_length)

    collator = Collator(tokenizer.pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)

    # Ottimizzatore
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) * args.num_train_epochs / max(1, args.gradient_accumulation_steps))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps) if args.linear_schedule else None

    global_step = 0
    best_eval_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(1, args.num_train_epochs + 1):
        print(f"\n[Epoch {epoch}] Inizio training...")
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Train E{epoch}")):
            source_ids = batch['source_ids'].to(device)
            source_mask = batch['source_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_mask = batch['target_mask'].to(device)

            loss, scaled_loss, active = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)
            (loss / args.gradient_accumulation_steps).backward()
            running_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    avg_loss = running_loss / args.logging_steps
                    print(f"[Step {global_step}] loss={avg_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")
                    running_loss = 0.0

        # Valutazione
        eval_loss = 0.0
        eval_steps = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Eval E{epoch}"):
                source_ids = batch['source_ids'].to(device)
                source_mask = batch['source_mask'].to(device)
                target_ids = batch['target_ids'].to(device)
                target_mask = batch['target_mask'].to(device)
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)
                eval_loss += loss.item()
                eval_steps += 1
        eval_loss = eval_loss / max(1, eval_steps)
        print(f"[Epoch {epoch}] Eval loss: {eval_loss:.4f}")

        # Early stopping
        if eval_loss < best_eval_loss - args.min_delta:
            best_eval_loss = eval_loss
            patience_counter = 0
            save_checkpoint(model, tokenizer, args.output_dir, step=global_step)
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                print("[EarlyStopping] Pazienza esaurita. Interrompo training.")
                break
        model.train()

    # Salvataggio finale
    save_checkpoint(model, tokenizer, args.output_dir, step=None)

    # (Opzionale) Test funzionale
    if args.do_functional_eval:
        print("\n[Valutazione Funzionale]")
        metrics = evaluate_functional(model, tokenizer, eval_ds, device, args.beam_size, args.max_target_length, pass_ks=(1,10), timeout=args.eval_timeout)
        print("Metriche:", metrics)
        with open(os.path.join(args.output_dir, "functional_eval.json"), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

    print("[Fine] Training completato.")

# ==========================================================
# Test helpers (caricamento e generazione rapida)
# ==========================================================

def load_trained_model(model_dir: str, beam_size: int, max_target_length: int, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Cerchiamo meta.json per sapere quale pretrained base usare
    meta_path = os.path.join(model_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        base_model_name = meta.get("pretrained_model", "microsoft/codebert-base")
        # Se l'utente non specifica beam_size/max_target_length usiamo quelli salvati (se coerente)
        if "max_target_length" in meta and max_target_length != meta["max_target_length"]:
            print(f"[Info] Override max_target_length richiesto: {max_target_length} (meta: {meta['max_target_length']})")
    else:
        base_model_name = "microsoft/codebert-base"
        print(f"[Warn] meta.json non trovato in {model_dir}; uso base model di default {base_model_name}. Creare meta.json se diverso.")

    # Carichiamo config del modello base (non quella locale che potrebbe mancare di pesi encoder HF)
    try:
        config = AutoConfig.from_pretrained(base_model_name)
        encoder = RobertaModel.from_pretrained(base_model_name, config=config)
    except Exception as e:
        raise RuntimeError(f"Impossibile ricostruire encoder dal modello base '{base_model_name}': {e}")

    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    sos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config, beam_size=beam_size, max_length=max_target_length, sos_id=sos_id, eos_id=eos_id)
    ckpt_files = [f for f in os.listdir(model_dir) if f.startswith('model_') and f.endswith('.pt')]
    if not ckpt_files:
        raise FileNotFoundError("Nessun checkpoint trovato in " + model_dir)
    # Prende il 'final' se esiste
    final = [f for f in ckpt_files if 'final' in f]
    ckpt_path = os.path.join(model_dir, final[0] if final else sorted(ckpt_files)[-1])
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"[Info] Caricato checkpoint {ckpt_path}")
    return model, tokenizer


def quick_test(model_dir: str, data_file: str, max_source_length: int = 512, max_target_length: int = 128, beam_size: int = 5, batch_size: int = 4):
    """Funzione rapida per generare output sulle prime N (batch_size) istanze del dataset.
    Restituisce lista di tuple (task_id, completion).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_trained_model(model_dir, beam_size, max_target_length, device)
    raw = read_humaneval(data_file)
    examples = [Example(task_id=r.get('task_id', f"task_{i}"), source=r['prompt'], target=r['canonical_solution']) for i,r in enumerate(raw)]
    subset = examples[:batch_size]
    ds = HumanEvalDataset(subset, tokenizer, max_source_length, max_target_length)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=Collator(tokenizer.pad_token_id))
    model.eval()
    results = []
    with torch.no_grad():
        for batch in dl:
            source_ids = batch['source_ids'].to(device)
            source_mask = batch['source_mask'].to(device)
            beam_outputs = model(source_ids=source_ids, source_mask=source_mask)
            seq = beam_outputs[0,0].tolist()
            if tokenizer.eos_token_id in seq:
                seq = seq[:seq.index(tokenizer.eos_token_id)]
            completion = tokenizer.decode(seq, skip_special_tokens=True)
            results.append((batch['task_ids'][0], completion))
    return results

# ==========================================================
# Argparse
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tuning Seq2Seq su HumanEval")
    p.add_argument('--data_file', type=str, required=True, help='Path al file HumanEval jsonl o jsonl.gz')
    p.add_argument('--pretrained_model', type=str, default='microsoft/codebert-base')
    p.add_argument('--output_dir', type=str, default='./finetuned_model')
    p.add_argument('--max_source_length', type=int, default=512)
    p.add_argument('--max_target_length', type=int, default=128)
    p.add_argument('--train_batch_size', type=int, default=8)
    p.add_argument('--eval_batch_size', type=int, default=8)
    p.add_argument('--learning_rate', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--num_train_epochs', type=int, default=3)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--beam_size', type=int, default=5)
    p.add_argument('--logging_steps', type=int, default=50)
    p.add_argument('--linear_schedule', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--early_stopping_patience', type=int, default=0, help='0 = disabilitato')
    p.add_argument('--min_delta', type=float, default=0.0, help='Miglioramento minimo per reset pazienza')
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--eval_split', type=float, default=0.2, help='Frazione (0-1) di dati per eval')
    p.add_argument('--do_functional_eval', action='store_true', help='Esegue valutazione pass@k se possibile')
    p.add_argument('--eval_timeout', type=float, default=3.0, help='Timeout in secondi per ogni test HumanEval')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
