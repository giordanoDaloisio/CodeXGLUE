#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs_llama/train_distill_%j.out
#SBATCH -J def_cuda
#SBATCH -p cuda
#SBATCH -c 10
#SBATCH --gres=gpu:4c_s80g:1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex


output_dir=./saved_models_llama_distill

cd code
srun python llama_distillation.py --output_dir=$output_dir