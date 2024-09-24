#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs/train_stud.out
#SBATCH -J tds
#SBATCH -p cuda
#SBATCH -c 40
#SBATCH --gres=gpu:large

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

base_model=microsoft/codebert-base
model_type=roberta
output_dir=./saved_models_distil_ase_3

cd code/compress
python distill.py \
    --do_train \
    --train_data_file ../../dataset/train_stud.jsonl \
    --eval_data_file ../../dataset/valid.jsonl \
    --model_dir ../saved_models_distil_ase3/checkpoint-best-mrr \
    --size 3 \
    --attention_heads 8 \
    --hidden_dim 96 \
    --intermediate_size 64 \
    --n_layers 12 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 20 \
    --seed 123456 2>&1
