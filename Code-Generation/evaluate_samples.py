#!/usr/bin/env python3
"""
Script per valutare i risultati di generazione del codice utilizzando HumanEval.
Calcola le metriche pass@1, pass@10, e pass@20.
"""

import json
import os
import argparse
import gzip
from typing import Dict, List, Optional
from collections import defaultdict, Counter
import numpy as np
import tqdm

from human_eval.data import read_problems, stream_jsonl, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness, estimate_pass_at_k
from human_eval.execution import check_correctness
from concurrent.futures import ThreadPoolExecutor, as_completed


def count_samples_per_task(sample_file: str) -> Dict[str, int]:
    """Conta il numero di campioni per ogni task nel file samples.jsonl"""
    task_counts = Counter()
    
    print(f"Analizzando il file {sample_file}...")
    for sample in stream_jsonl(sample_file):
        task_id = sample["task_id"]
        task_counts[task_id] += 1
    
    return dict(task_counts)


def validate_samples_for_passat_k(sample_file: str, k_values: List[int]) -> bool:
    """Verifica se ci sono abbastanza campioni per calcolare pass@k"""
    task_counts = count_samples_per_task(sample_file)
    
    min_samples = min(task_counts.values()) if task_counts else 0
    max_k = max(k_values)
    
    print(f"\nStatistiche campioni:")
    print(f"- Numero di task: {len(task_counts)}")
    print(f"- Minimo campioni per task: {min_samples}")
    print(f"- Massimo campioni per task: {max(task_counts.values()) if task_counts else 0}")
    print(f"- K massimo richiesto: {max_k}")
    
    if min_samples < max_k:
        print(f"\n⚠️  Attenzione: Non ci sono abbastanza campioni per calcolare pass@{max_k}")
        print(f"   Alcuni task hanno solo {min_samples} campioni")
        
        # Filtra i valori k che sono supportati
        valid_k = [k for k in k_values if k <= min_samples]
        if valid_k:
            print(f"   Si possono calcolare solo: {valid_k}")
        return False, valid_k
    
    return True, k_values


def evaluate_samples(
    sample_file: str,
    k_values: List[int] = [1, 10, 20],
    n_workers: int = 4,
    timeout: float = 3.0,
    output_dir: str = None
) -> Dict[str, float]:
    """
    Valuta la correttezza funzionale dei campioni generati e calcola pass@k.
    
    Args:
        sample_file: Path al file samples.jsonl
        k_values: Lista dei valori k per cui calcolare pass@k
        n_workers: Numero di worker per l'esecuzione parallela
        timeout: Timeout per l'esecuzione di ogni test
        output_dir: Directory dove salvare i risultati dettagliati
        
    Returns:
        Dizionario con le metriche pass@k
    """
    
    # Verifica se il file esiste
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"File non trovato: {sample_file}")
    
    # Valida i campioni
    is_valid, valid_k = validate_samples_for_passat_k(sample_file, k_values)
    if not is_valid:
        print(f"\n⚠️  Utilizzando valori k validi: {valid_k}")
        k_values = valid_k
        
    if not k_values:
        raise ValueError("Non ci sono abbastanza campioni per calcolare alcun pass@k")
    
    print(f"\n🚀 Iniziando valutazione con pass@{k_values}...")
    
    # Carica i problemi HumanEval
    problems = read_problems()
    
    # Esegue la valutazione utilizzando la funzione di HumanEval
    results = evaluate_functional_correctness(
        sample_file=sample_file,
        k=k_values,
        n_workers=n_workers,
        timeout=timeout
    )
    
    # Salva risultati dettagliati se richiesto
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"evaluation_results.json")
        
        # Aggiungi metadati
        detailed_results = {
            "pass_at_k": results,
            "metadata": {
                "sample_file": sample_file,
                "k_values": k_values,
                "n_workers": n_workers,
                "timeout": timeout,
                "total_problems": len(problems)
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n📁 Risultati dettagliati salvati in: {results_file}")
    
    return results


def print_results(results: Dict[str, float], sample_file: str):
    """Stampa i risultati in formato leggibile"""
    
    print(f"\n{'='*50}")
    print(f"🎯 RISULTATI VALUTAZIONE - {os.path.basename(sample_file)}")
    print(f"{'='*50}")
    
    for metric, value in sorted(results.items()):
        percentage = value * 100
        print(f"{metric:>10}: {percentage:6.2f}%")
    
    # Trova il miglior risultato
    if results:
        best_metric = max(results.items(), key=lambda x: x[1])
        print(f"\n🏆 Miglior risultato: {best_metric[0]} = {best_metric[1]*100:.2f}%")
    
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Valuta i risultati di generazione del codice usando HumanEval"
    )
    parser.add_argument(
        "sample_file",
        type=str,
        help="Path al file samples.jsonl da valutare"
    )
    parser.add_argument(
        "--k",
        type=str,
        default="1,10,20",
        help="Valori k per pass@k separati da virgola (default: 1,10,20)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Numero di worker per esecuzione parallela (default: 4)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Timeout per esecuzione test in secondi (default: 3.0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory dove salvare risultati dettagliati (opzionale)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Modalità silenziosa - mostra solo i risultati finali"
    )

    args = parser.parse_args()
    
    # Parse dei valori k
    try:
        k_values = [int(k.strip()) for k in args.k.split(",")]
        k_values = sorted(set(k_values))  # Rimuove duplicati e ordina
    except ValueError:
        print("❌ Errore: I valori k devono essere numeri interi separati da virgola")
        return 1
    
    try:
        # Esegue la valutazione
        results = evaluate_samples(
            sample_file=args.sample_file,
            k_values=k_values,
            n_workers=args.n_workers,
            timeout=args.timeout,
            output_dir=args.output_dir
        )
        
        # Stampa i risultati
        if not args.quiet:
            print_results(results, args.sample_file)
        else:
            # Modalità silenziosa - solo risultati
            for metric, value in sorted(results.items()):
                print(f"{metric}: {value:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Errore durante la valutazione: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
