#!/bin/bash
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs/kd_distillation_%j.out
#SBATCH -J kd_distill
#SBATCH -p cuda
#SBATCH -c 20
#SBATCH --gres=gpu:fat

# Script per Knowledge Distillation di CodeGPT su DistilGPT-2
# per code generation su CodeSearchNet-Python
# Uso: sbatch run_kd_distillation.sh
# oppure: bash run_kd_distillation.sh (senza SLURM)

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

# ========================================
# Parametri Models
# ========================================
TEACHER_MODEL="./codegen_finetuned"  # Teacher (grande, accurato)
STUDENT_MODEL="distilbert/distilgpt2"        # Student (piccolo, veloce)
OUTPUT_DIR="./distilgpt_distilled"

# ========================================
# Parametri Distillation
# ========================================
TEMPERATURE=2.0    # Temperature per softmax (2.0-4.0 tipico per distillation)
ALPHA=0.5          # Bilancia CE loss (alpha) vs KL loss (1-alpha)
                   # 0.5 = bilanciato, 0.7 = più peso a ground truth, 0.3 = più peso a teacher

# ========================================
# Parametri Training
# ========================================
NUM_EPOCHS=3
BATCH_SIZE=6       # Ridotto per evitare OOM (2 modelli in memoria)
LEARNING_RATE=5e-5 # Learning rate per student
MAX_SEQ_LENGTH=512

# ========================================
# Parametri Dataset (opzionali per test veloce)
# ========================================
# Decommentare per limitare dataset:
# MAX_TRAIN_SAMPLES=10000
# MAX_EVAL_SAMPLES=1000

# ========================================
# Parametri HumanEval
# ========================================
HUMANEVAL_FILE="../data/HumanEval.jsonl.gz"
HUMANEVAL_NUM_SAMPLES=20
HUMANEVAL_MAX_TOKENS=256
HUMANEVAL_TEMPERATURE=0.8

# ========================================
# Logging
# ========================================
LOGGING_STEPS=100
SAVE_STEPS=2000
EVAL_STEPS=2000

echo "========================================"
echo "Knowledge Distillation per Code Generation"
echo "========================================"
echo "Teacher: $TEACHER_MODEL"
echo "Student: $STUDENT_MODEL"
echo "Output: $OUTPUT_DIR"
echo "----------------------------------------"
echo "Temperature: $TEMPERATURE"
echo "Alpha (CE weight): $ALPHA"
echo "1-Alpha (KL weight): $(echo "1 - $ALPHA" | bc)"
echo "----------------------------------------"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "========================================"

# Crea directory logs se non esiste
mkdir -p ./logs
cd code

# Esegui Knowledge Distillation
python distillation.py \
    --teacher_model_path $TEACHER_MODEL \
    --student_model_path $STUDENT_MODEL \
    --output_dir $OUTPUT_DIR \
    --temperature $TEMPERATURE \
    --alpha $ALPHA \
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
    --humaneval_num_samples $HUMANEVAL_NUM_SAMPLES \
    --humaneval_max_tokens $HUMANEVAL_MAX_TOKENS \
    --humaneval_temperature $HUMANEVAL_TEMPERATURE \
    --seed 42

# Se hai specificato MAX_TRAIN_SAMPLES, aggiungili:
# ${MAX_TRAIN_SAMPLES:+--max_train_samples $MAX_TRAIN_SAMPLES} \
# ${MAX_EVAL_SAMPLES:+--max_eval_samples $MAX_EVAL_SAMPLES} \

echo ""
echo "========================================"
echo "Distillation completata!"
echo "Student model salvato in: $OUTPUT_DIR"
echo "Risultati HumanEval: $OUTPUT_DIR/humaneval_results/results.json"
echo "========================================"
echo ""
echo "Suggerimenti per ottimizzare:"
echo "- Aumenta TEMPERATURE (3.0-4.0) per distillation più soft"
echo "- Riduci ALPHA (0.3-0.4) per dare più peso alla KL loss"
echo "- Aumenta NUM_EPOCHS se student underfitta"
echo "========================================"
