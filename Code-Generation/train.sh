#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs/codegen_train.out
#SBATCH -J codegen
#SBATCH -p cuda
#SBATCH -c 40
#SBATCH --gres=gpu:fat

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd code
lang=java #programming language
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=../data/HumanEval.jsonl.gz
output_dir=model
train_file=$data_dir
dev_file=$data_dir
epochs=3
pretrained_model=microsoft/codebert-base

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

srun python finetune_humaneval.py --data_file $data_dir
