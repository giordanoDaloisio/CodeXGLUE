# 📊 Script di Valutazione HumanEval

Questo set di script permette di valutare i risultati di generazione del codice utilizzando il benchmark HumanEval, calcolando le metriche **pass@1**, **pass@10** e **pass@20**.

## 🗂 File Inclusi

- **`evaluate_samples.py`** - Script principale completo e configurabile
- **`quick_evaluate.py`** - Script semplificato per valutazioni rapide  
- **`run_evaluation.sh`** - Script bash per automazione completa
- **`EVALUATION_README.md`** - Questa guida

## 🚀 Utilizzo Rapido

### Opzione 1: Script Semplificato (Raccomandato)

```bash
# Valutazione automatica del file samples.jsonl
python3 quick_evaluate.py

# Valutazione di un file specifico
python3 quick_evaluate.py samples_123.jsonl
```

### Opzione 2: Script Bash Automatico

```bash
# Trova e valuta automaticamente i file samples
./run_evaluation.sh

# Valuta un file specifico
./run_evaluation.sh samples_123.jsonl

# Con parametri personalizzati
./run_evaluation.sh samples_123.jsonl --k 1,5,10,20
```

### Opzione 3: Script Python Completo

```bash
# Valutazione standard
python3 evaluate_samples.py samples.jsonl

# Con parametri personalizzati
python3 evaluate_samples.py samples.jsonl --k 1,10,20 --n_workers 8 --timeout 5.0

# Con salvataggio risultati dettagliati
python3 evaluate_samples.py samples.jsonl --output_dir my_results/
```

## 📋 Parametri Disponibili

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `--k` | Valori k per pass@k (es: "1,10,20") | "1,10,20" |
| `--n_workers` | Numero di processi paralleli | 4 |
| `--timeout` | Timeout per ogni test (secondi) | 3.0 |
| `--output_dir` | Directory per risultati dettagliati | - |
| `--quiet` | Modalità silenziosa | False |

## 📊 Esempio di Output

```
==================================================
🎯 RISULTATI VALUTAZIONE - samples_123.jsonl
==================================================
    pass@1:  65.85%
   pass@10:  82.93%
   pass@20:  89.63%

🏆 Miglior risultato: pass@20 = 89.63%
==================================================
```

## 📁 Formato File di Input

Il file `samples.jsonl` deve avere il formato:

```jsonl
{"task_id": "HumanEval/0", "completion": "def has_close_elements..."}
{"task_id": "HumanEval/1", "completion": "def separate_paren_groups..."}
```

## 🔧 Requisiti

Gli script utilizzano la libreria HumanEval già presente nel progetto:

```
human_eval/
├── data.py
├── evaluation.py  
├── execution.py
└── evaluate_functional_correctness.py
```

## 📈 Metriche Calcolate

- **pass@k**: Probabilità che almeno una delle prime k soluzioni sia corretta
- La metrica è calcolata usando la formula di stima: `1 - comb(n-c, k) / comb(n, k)`
  - `n` = numero totale di campioni per task
  - `c` = numero di campioni corretti per task  
  - `k` = numero di tentativi considerati

## ⚠️ Note Importanti

1. **Campioni necessari**: Per calcolare pass@k serve almeno k campioni per ogni task
2. **Tempo di esecuzione**: La valutazione può richiedere diversi minuti per file grandi
3. **File di output**: 
   - `{sample_file}_results.jsonl` - Risultati dettagliati per ogni campione
   - `evaluation_results.json` - Riassunto delle metriche (se specificato --output_dir)

## 🐛 Troubleshooting

### Errore "Not enough samples"
```
⚠️ Attenzione: Non ci sono abbastanza campioni per calcolare pass@20
   Alcuni task hanno solo 10 campioni
   Si possono calcolare solo: [1, 10]
```
**Soluzione**: Genera più campioni o usa valori k più bassi

### File non trovato
```
❌ File non trovato: samples.jsonl
```
**Soluzione**: Verifica che il file esista e sia nel percorso corretto

### Timeout errori
```
❌ Errore durante la valutazione: Timeout
```
**Soluzione**: Aumenta il timeout con `--timeout 10.0`

## 🔄 Integrazione con Altri Script

Gli script possono essere facilmente integrati nel workflow esistente:

```bash
# Genera samples
python3 generation.py --job_id 123 --num_samples_per_task 20

# Valuta automaticamente
python3 quick_evaluate.py samples_123.jsonl
```

## 📞 Support

Per problemi o domande:
1. Controlla che i file samples.jsonl siano nel formato corretto
2. Verifica che la libreria human_eval sia installata correttamente  
3. Controlla i log per errori specifici
