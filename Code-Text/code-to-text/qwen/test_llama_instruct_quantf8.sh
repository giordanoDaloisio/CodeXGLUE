#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs_qwen/cuda_instruct_quantf8_%j.out
#SBATCH -J llama_qwen
#SBATCH -p cuda
#SBATCH -c 20
# SBATCH --gres=gpu:3c_s80g:1
#SBATCH --gres=gpu:large


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd code
source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

srun python code_summarization_llama.py --quantf8 --job_id=$SLURM_JOB_ID --model_name Qwen/Qwen3-4B-Instruct-2507
