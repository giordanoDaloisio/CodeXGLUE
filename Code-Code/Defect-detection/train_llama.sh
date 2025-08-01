#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs_llama/train_%j.out
#SBATCH -J defect
#SBATCH -p cuda
#SBATCH -c 10
#SBATCH --gres=gpu:3c_s80g:1

# SBATCH --gres=gpu:large:4
# SBATCH --ntasks-per-node=4

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

output_dir=./saved_models_llama

cd code
srun python run_lora_simple.py 
