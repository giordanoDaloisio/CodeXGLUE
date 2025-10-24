#!/bin/bash
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs/distilgpt_finetune_%j.out
#SBATCH -J distilgpt_ft
#SBATCH -p cuda
#SBATCH -c 20
#SBATCH --gres=gpu:fat

# Script per fine-tuning DistilGPT su CodeSearchNet-Python e valutazione su HumanEval
# Uso: sbatch run_distilgpt_finetune.sh
# oppure: bash run_distilgpt_finetune.sh (senza SLURM)

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
MODEL_NAME="distilbert/distilgpt2"  # CAMBIATO: usa modello trainato su codice
OUTPUT_DIR="./codegen_finetuned"
HUMANEVAL_FILE="../data/HumanEval.jsonl.gz"

NUM_EPOCHS=3
BATCH_SIZE=6  # Ridotto per CodeGen più grande (evita OOM)
LEARNING_RATE=2e-5  # Più basso per modello pre-trained
MAX_SEQ_LENGTH=512

# Parametri opzionali per debug/test veloce
# Decommentare per limitare dataset (utile per test):
# MAX_TRAIN_SAMPLES=1000
# MAX_EVAL_SAMPLES=200

# Per test rapido (rimuovi # per abilitare):
# MAX_TRAIN_SAMPLES=10000
# MAX_EVAL_SAMPLES=1000

# Logging
LOGGING_STEPS=100
SAVE_STEPS=2000
EVAL_STEPS=2000

echo "========================================"
echo "Fine-tuning CodeGPT su CodeSearchNet-Python"
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
python finetune_codegpt_codesearchnet.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --do_eval \
    --eval_on_humaneval \
    --humaneval_file $HUMANEVAL_FILE \
    --num_train_epochs $NUM_EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --eval_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --gradient_accumulation_steps 2 \
    --warmup_steps 1000 \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_total_limit 2 \
    --fp16 \
    --humaneval_num_samples 20 \
    --humaneval_max_tokens 256 \
    --humaneval_temperature 0.6 \
    --seed 42

# Nota: rimuovere --fp16 se non hai GPU o da errore

echo ""
echo "========================================"
echo "Training completato!"
echo "Modello salvato in: $OUTPUT_DIR"
echo "Risultati HumanEval: $OUTPUT_DIR/humaneval_results/results.json"
echo "========================================"
