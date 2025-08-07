#!/usr/bin/env python3
"""
Script semplificato per valutare rapidamente i file samples.jsonl
con pass@1, pass@10, e pass@20.
"""

import os
import sys
import glob
from typing import List

# Aggiungi la directory corrente al path per importare human_eval
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_samples import evaluate_samples, print_results


def find_sample_files() -> List[str]:
    """Trova tutti i file samples*.jsonl nella directory corrente"""
    pattern = "samples*.jsonl"
    files = glob.glob(pattern)
    
    # Escludi i file di risultati
    files = [f for f in files if not f.endswith("_results.jsonl")]
    
    return sorted(files)


def quick_evaluate(sample_file: str):
    """Valutazione rapida con parametri standard"""
    print(f"\n🔍 Valutando: {sample_file}")
    
    try:
        # Usa parametri standard: pass@1, pass@10, pass@20
        results = evaluate_samples(
            sample_file=sample_file,
            k_values=[1, 10, 20],
            n_workers=4,
            timeout=3.0,
            output_dir=f"results/{os.path.splitext(sample_file)[0]}"
        )
        
        print_results(results, sample_file)
        return results
        
    except Exception as e:
        print(f"❌ Errore durante la valutazione di {sample_file}: {e}")
        return None


def main():
    """Main function per valutazione rapida"""
    
    # Se fornito un file specifico come argomento
    if len(sys.argv) > 1:
        sample_file = sys.argv[1]
        if not os.path.exists(sample_file):
            print(f"❌ File non trovato: {sample_file}")
            sys.exit(1)
        
        results = quick_evaluate(sample_file)
        if results is None:
            sys.exit(1)
        return
    
    # Altrimenti cerca automaticamente i file
    sample_files = find_sample_files()
    
    if not sample_files:
        print("❌ Nessun file samples*.jsonl trovato nella directory corrente!")
        print("\nUsage:")
        print("  python3 quick_evaluate.py [sample_file.jsonl]")
        print("\nOppure posizionati in una directory che contiene file samples*.jsonl")
        sys.exit(1)
    
    print(f"🔍 Trovati {len(sample_files)} file da valutare:")
    for i, file in enumerate(sample_files, 1):
        print(f"  {i}. {file}")
    
    # Se c'è solo un file, valutalo direttamente
    if len(sample_files) == 1:
        print(f"\n🚀 Valutando automaticamente l'unico file trovato...")
        results = quick_evaluate(sample_files[0])
        if results is None:
            sys.exit(1)
        return
    
    # Se ci sono più file, chiedi all'utente di scegliere
    print(f"\n📋 Scegli quale file valutare:")
    print(f"  0. Tutti i file")
    for i, file in enumerate(sample_files, 1):
        print(f"  {i}. {file}")
    
    try:
        choice = input("\nInserisci il numero della tua scelta (0 per tutti): ").strip()
        
        if choice == "0":
            # Valuta tutti i file
            print(f"\n🚀 Valutando tutti i {len(sample_files)} file...")
            all_results = {}
            
            for sample_file in sample_files:
                results = quick_evaluate(sample_file)
                if results:
                    all_results[sample_file] = results
            
            # Mostra riassunto finale
            if all_results:
                print(f"\n" + "="*60)
                print(f"📊 RIASSUNTO FINALE - {len(all_results)} file valutati")
                print(f"="*60)
                
                for sample_file, results in all_results.items():
                    print(f"\n📁 {sample_file}:")
                    for metric, value in sorted(results.items()):
                        print(f"  {metric:>10}: {value*100:6.2f}%")
            
        else:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(sample_files):
                selected_file = sample_files[choice_idx]
                results = quick_evaluate(selected_file)
                if results is None:
                    sys.exit(1)
            else:
                print("❌ Scelta non valida!")
                sys.exit(1)
                
    except (ValueError, KeyboardInterrupt):
        print("\n❌ Operazione annullata!")
        sys.exit(1)


if __name__ == "__main__":
    main()
