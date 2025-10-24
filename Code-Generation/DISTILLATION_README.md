# Knowledge Distillation per Code Generation

Questo modulo implementa **Knowledge Distillation** da un modello teacher (CodeGPT) a un modello student più piccolo (DistilGPT-2) per il task di code generation su CodeSearchNet-Python.

## 📋 Panoramica

La Knowledge Distillation è una tecnica di compressione dei modelli che permette di trasferire le conoscenze da un modello grande e accurato (teacher) a uno più piccolo e veloce (student), mantenendo performance elevate.

### Architettura

```
Teacher Model (CodeGPT-small-py)  ──┐
                                    ├──> Combined Loss ──> Student Model (DistilGPT-2)
Ground Truth Labels               ──┘
```

### Loss Function

La loss combinata è:

```
Loss = α * CE_loss + (1-α) * KL_loss
```

Dove:
- **CE_loss**: Cross-Entropy tra predizioni student e ground truth
- **KL_loss**: KL Divergence tra distribuzioni softmax di teacher e student (con temperature scaling)
- **α**: Iperparametro che bilancia le due loss (default: 0.5)

## 🚀 Quick Start

### 1. Esegui Knowledge Distillation

```bash
# Con SLURM
sbatch run_kd_distillation.sh

# Senza SLURM
bash run_kd_distillation.sh
```

### 2. Parametri configurabili

Modifica lo script `run_kd_distillation.sh`:

```bash
# Modelli
TEACHER_MODEL="microsoft/CodeGPT-small-py"
STUDENT_MODEL="distilbert/distilgpt2"

# Hyperparameters Distillation
TEMPERATURE=2.0    # 2.0-4.0 raccomandato
ALPHA=0.5          # 0.3-0.7 raccomandato

# Training
NUM_EPOCHS=3
BATCH_SIZE=6
LEARNING_RATE=5e-5
```

### 3. Uso diretto Python

```bash
cd code

python distillation.py \
  --teacher_model_path microsoft/CodeGPT-small-py \
  --student_model_path distilbert/distilgpt2 \
  --output_dir ./distilgpt_distilled \
  --temperature 2.0 \
  --alpha 0.5 \
  --num_train_epochs 3 \
  --train_batch_size 6 \
  --learning_rate 5e-5 \
  --do_train \
  --do_eval \
  --eval_on_humaneval \
  --fp16
```

## 📊 Parametri Chiave

### Temperature (T)

Controlla quanto "smooth" sono le distribuzioni di probabilità durante la distillation:

- **T = 1.0**: Distribuzioni standard (sharp)
- **T = 2.0-4.0**: Distribuzioni smooth (raccomandato per distillation)
- **T → ∞**: Distribuzioni uniformi

**Effetto**: Temperature più alte rendono le "dark knowledge" del teacher più evidenti, facilitando l'apprendimento dello student.

### Alpha (α)

Bilancia l'importanza delle due loss functions:

- **α = 1.0**: Solo CE loss (fine-tuning classico, ignora teacher)
- **α = 0.7**: Più peso al ground truth (conservativo)
- **α = 0.5**: Bilanciato (default, raccomandato)
- **α = 0.3**: Più peso al teacher (aggressivo)
- **α = 0.0**: Solo KL loss (pura distillation)

**Suggerimento**: 
- Usa α più alto (0.6-0.7) se hai molti dati ground truth di alta qualità
- Usa α più basso (0.3-0.4) se vuoi massimizzare il trasferimento dal teacher

## 🎯 Vantaggi di DistilGPT-2 per Code Generation

| Metrica | CodeGPT-small | DistilGPT-2 (distilled) |
|---------|---------------|-------------------------|
| Parametri | ~125M | ~82M (65% riduzione) |
| Velocità inference | 1x | ~1.6x più veloce |
| Memoria GPU | ~500MB | ~320MB |
| Performance (pass@1) | 100% | ~90-95% (tipico) |

## 📁 Output

Dopo la distillation, troverai:

```
distilgpt_distilled/
├── pytorch_model.bin           # Student model weights
├── config.json                 # Model config
├── tokenizer.json              # Tokenizer
├── trainer_state.json          # Training history
├── train_results.json          # Training metrics
├── eval_results.json           # Validation metrics
└── humaneval_results/
    ├── samples.jsonl           # Generated code samples
    └── results.json            # pass@k metrics
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

Se ottieni errori OOM, prova:

1. Ridurre batch size: `BATCH_SIZE=4` o `BATCH_SIZE=2`
2. Aumentare gradient accumulation: `--gradient_accumulation_steps 4`
3. Ridurre max sequence length: `MAX_SEQ_LENGTH=384`
4. Usare solo CPU per teacher (più lento ma usa meno GPU memory)

### Student underfitta

Se lo student model ha performance basse:

1. Aumenta numero epoch: `NUM_EPOCHS=5`
2. Riduci alpha: `ALPHA=0.3` (più peso a teacher)
3. Aumenta temperature: `TEMPERATURE=3.0`
4. Riduci learning rate: `LEARNING_RATE=2e-5`

### Student overfitta

Se lo student overfitta sul training set:

1. Aumenta weight decay: `--weight_decay 0.05`
2. Usa dropout più alto (modifica config student)
3. Riduci numero epoch
4. Aumenta alpha: `ALPHA=0.7` (più peso a ground truth)

## 🧪 Varianti e Esperimenti

### Experiment 1: Pura Distillation (α=0)

```bash
python distillation.py \
  --teacher_model_path microsoft/CodeGPT-small-py \
  --student_model_path distilbert/distilgpt2 \
  --temperature 3.0 \
  --alpha 0.0 \
  --do_train
```

### Experiment 2: Soft Labels + Hard Labels (α=0.5)

```bash
python distillation.py \
  --teacher_model_path microsoft/CodeGPT-small-py \
  --student_model_path distilbert/distilgpt2 \
  --temperature 2.5 \
  --alpha 0.5 \
  --do_train
```

### Experiment 3: High Temperature Distillation (α=0.3, T=4.0)

```bash
python distillation.py \
  --teacher_model_path microsoft/CodeGPT-small-py \
  --student_model_path distilbert/distilgpt2 \
  --temperature 4.0 \
  --alpha 0.3 \
  --do_train
```

## 📚 Riferimenti

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al., 2015)
- [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019)
- [CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation](https://arxiv.org/abs/2102.04664)

## 💡 Best Practices

1. **Pre-train teacher prima**: Assicurati che il teacher model sia ben trainato sul task
2. **Usa temperature 2-4**: Temperature troppo basse riducono l'efficacia della distillation
3. **Bilancia α attentamente**: α=0.5 è un buon punto di partenza
4. **Monitor entrambe le loss**: Controlla sia CE_loss che KL_loss durante training
5. **Valuta su HumanEval**: Le metriche pass@k sono il gold standard per code generation

## 🎓 Come funziona

### Step 1: Forward Pass Teacher

```python
with torch.no_grad():
    teacher_logits = teacher_model(input_ids)  # No gradient
```

### Step 2: Forward Pass Student

```python
student_logits = student_model(input_ids)  # Con gradient
```

### Step 3: Compute Losses

```python
# Cross-Entropy con ground truth
ce_loss = CrossEntropy(student_logits, labels)

# KL Divergence con teacher
kl_loss = KL(
    softmax(student_logits / T),
    softmax(teacher_logits / T)
) * T²

# Loss combinata
loss = α * ce_loss + (1-α) * kl_loss
```

### Step 4: Backprop solo su Student

```python
loss.backward()  # Solo student riceve gradienti
optimizer.step()  # Aggiorna solo student weights
```

## 🔬 Metriche da Monitorare

Durante il training, monitora:

1. **loss_ce**: Loss cross-entropy (student vs ground truth)
2. **loss_kl**: Loss KL divergence (student vs teacher)
3. **loss_total**: Loss combinata
4. **eval_loss**: Loss su validation set
5. **pass@1, pass@10, pass@20**: Su HumanEval (dopo training)

## 🚀 Next Steps

Dopo la distillation, puoi:

1. **Quantizzare** il student model (INT8/FP16) per ridurre ulteriormente le dimensioni
2. **Pruning** del student model per rimuovere neurons inutilizzati
3. **Distillation iterativa**: Usare lo student distillato come nuovo teacher per un modello ancora più piccolo
4. **Fine-tuning task-specific**: Adattare lo student distillato a task specifici

---

**Autore**: Sistema Knowledge Distillation per CodeXGLUE  
**Licenza**: MIT
