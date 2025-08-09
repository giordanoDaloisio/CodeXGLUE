# Code Summarization with LLaMA-3.1-8B Instruct

Questo progetto implementa un sistema di code summarization utilizzando il modello LLaMA-3.1-8B Instruct con few-shot prompting. Il sistema utilizza il dataset Java del CodeXGLUE per validation e testing.

## Caratteristiche

- **Modello**: LLaMA-3.1-8B Instruct
- **Tecnica**: Few-shot prompting con esempi dal dataset di validation
- **Dataset**: CodeXGLUE Code-to-Text (Java)
- **Metriche**: BLEU, METEOR, ROUGE-1, ROUGE-2, ROUGE-L
- **Ottimizzazioni**: Quantizzazione 8-bit per ridurre l'uso di memoria

## Struttura dei File

- `code_summarization_llama.py`: Script principale per la summarization
- `quick_test_summarization.py`: Test rapido con pochi esempi
- `evaluate_results.py`: Script per calcolare le metriche di valutazione
- `setup_summarization.sh`: Script di setup dell'ambiente
- `requirements_summarization.txt`: Dipendenze Python
- `README_SUMMARIZATION.md`: Questa guida

## Setup

1. **Installa le dipendenze**:
   ```bash
   ./setup_summarization.sh
   ```

2. **Verifica che i dati siano disponibili**:
   - Validation: `/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/valid.jsonl`
   - Test: `/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/test.jsonl`

## Utilizzo

### 1. Test Rapido (5 esempi)
```bash
python3 quick_test_summarization.py
```

### 2. Valutazione Completa
```bash
# Valutazione su 100 campioni di test con 3 few-shot examples
python3 code_summarization_llama.py --num_test_samples 100 --num_few_shot 3

# Valutazione personalizzata
python3 code_summarization_llama.py \
    --num_test_samples 50 \
    --num_few_shot 5 \
    --output_file my_results.json \
    --temperature 0.2
```

### 3. Valutazione dei Risultati
```bash
# Calcola metriche base
python3 evaluate_results.py results.json

# Analisi dettagliata con esempi
python3 evaluate_results.py results.json --detailed --num_examples 10
```

## Parametri Principali

### `code_summarization_llama.py`
- `--validation_file`: File JSONL di validation (default: auto)
- `--test_file`: File JSONL di test (default: auto) 
- `--model_name`: Nome del modello HuggingFace (default: meta-llama/Meta-Llama-3.1-8B-Instruct)
- `--num_few_shot`: Numero di esempi few-shot (default: 3)
- `--num_test_samples`: Numero di campioni di test (default: 100)
- `--output_file`: File di output per i risultati
- `--no_quantization`: Disabilita quantizzazione 8-bit
- `--temperature`: Temperatura per il sampling (default: 0.1)

### `evaluate_results.py`
- `results_file`: File JSON con i risultati
- `--detailed`: Mostra analisi dettagliata
- `--num_examples`: Numero di esempi da mostrare

## Formato dei Dati

### Input (JSONL)
```json
{
  "repo": "ReactiveX/RxJava",
  "path": "src/main/java/io/reactivex/Observable.java", 
  "func_name": "Observable.wrap",
  "code": "@CheckReturnValue\npublic static <T> Observable<T> wrap(ObservableSource<T> source) {...}",
  "docstring": "Wraps an ObservableSource into an Observable if not already an Observable.",
  "language": "java"
}
```

### Output (JSON)
```json
{
  "repo": "ReactiveX/RxJava",
  "path": "src/main/java/io/reactivex/Observable.java",
  "func_name": "Observable.wrap", 
  "original_code": "...",
  "original_summary": "Wraps an ObservableSource into an Observable if not already an Observable.",
  "generated_summary": "Converts an ObservableSource to an Observable instance.",
  "url": "https://github.com/..."
}
```

## Few-Shot Prompting

Il sistema utilizza un prompt strutturato con esempi dal dataset di validation:

```
You are an expert software developer tasked with writing concise and accurate summaries for Java methods. 

Example 1:
Code:
```java
public long copyTo(CharSink sink) throws IOException {
    // implementation
}
```
Summary: Copies the contents of this source to the given sink.

[più esempi...]

Now, please provide a summary for this Java method:
Code:
```java
[codice da riassumere]
```
Summary:
```

## Metriche di Valutazione

- **BLEU**: Misura la precisione n-gram tra summary generate e di riferimento
- **METEOR**: Considera sinonimi e stemming
- **ROUGE-1**: Overlap di unigram
- **ROUGE-2**: Overlap di bigram  
- **ROUGE-L**: Longest Common Subsequence

## Requisiti di Sistema

- **GPU**: Consigliata (NVIDIA con CUDA) per prestazioni ottimali
- **RAM**: Almeno 16GB (8GB con quantizzazione)
- **VRAM**: Almeno 8GB per il modello completo, 4GB con quantizzazione
- **Spazio**: ~20GB per il modello e dipendenze

## Risoluzione Problemi

### Error: CUDA out of memory
```bash
# Abilita quantizzazione (default) o riduci batch size
python3 code_summarization_llama.py --num_test_samples 50
```

### Error: Model not found
Assicurati di avere accesso al modello LLaMA su HuggingFace e di aver effettuato il login:
```bash
pip install huggingface_hub
huggingface-cli login
```

### Error: NLTK data not found
```bash
python3 -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

## Esempi di Risultati

```
Original:  Returns an Observable that emits the events emitted by source ObservableSource in a sorted order.
Generated: Sorts the elements emitted by the source Observable using a specified comparison function.

BLEU: 0.234, ROUGE-1: 0.567, METEOR: 0.445
```

## Estensioni Possibili

1. **Altri linguaggi**: Python, JavaScript, ecc.
2. **Modelli diversi**: CodeT5, CodeBERT, altri LLaMA
3. **Fine-tuning**: Adattamento specifico per code summarization
4. **Prompt engineering**: Ottimizzazione dei prompt few-shot
5. **Metriche personalizzate**: Metriche specifiche per codice
