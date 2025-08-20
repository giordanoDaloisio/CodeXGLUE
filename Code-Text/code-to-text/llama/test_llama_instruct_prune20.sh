#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs_llama/cuda_instruct_prune20_%j.out
#SBATCH -J llama_test
#SBATCH -p cuda
#SBATCH -c 10
# SBATCH --gres=gpu:3c_s80g:1
#SBATCH --gres=gpu:fat


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd code
source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

srun python code_summarization_llama.py --prune20 --job_id=$SLURM_JOB_ID
