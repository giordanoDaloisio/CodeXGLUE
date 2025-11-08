#!/bin/bash
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs/codegpt_test_i8_cpu_%j.out
#SBATCH -J codegpt_test
#SBATCH -p normal
#SBATCH -w compute-0-7
#SBATCH -c 20

# Script per fine-tuning CodeGPT su CodeSearchNet-Python e valutazione su HumanEval
# Uso: sbatch run_codegpt_finetune.sh
# oppure: bash run_codegpt_finetune.sh (senza SLURM)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Attiva ambiente conda (modifica se necessario)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate codex
fi

# Parametri configurabili
MODEL_NAME="./codegen_finetuned"  # o gpt2, Salesforce/codegen-350M-mono
OUTPUT_DIR="./codegen_finetuned"
HUMANEVAL_FILE="../data/HumanEval.jsonl.gz"

NUM_EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=5e-5
MAX_SEQ_LENGTH=512

# Parametri opzionali per debug/test veloce
# Decommentare per limitare dataset (utile per test):
# MAX_TRAIN_SAMPLES=1000
# MAX_EVAL_SAMPLES=200

# Logging
LOGGING_STEPS=100
SAVE_STEPS=2000
EVAL_STEPS=2000

echo "========================================"
echo "Testing CodeGPT su HumanEval"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "========================================"

# Crea directory logs se non esiste
mkdir -p ./logs
cd code
# Esegui training + valutazione
python evaluate.py \
    --model_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --humaneval_file $HUMANEVAL_FILE \
    --quantize_i8 \

# Nota: rimuovere --fp16 se non hai GPU o da errore

echo ""
echo "========================================"
echo "Testing completato!"
echo "========================================"
