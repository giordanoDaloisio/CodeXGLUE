#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs/codegen_qwen_quant4_%j.out
#SBATCH -J codegen_qwen
#SBATCH -p cuda
#SBATCH -c 10
#SBATCH --gres=gpu:3c_s80g:1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}


source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

srun python generation.py --job_id $SLURM_JOB_ID --quant4 --model_name_or_path Qwen/Qwen2.5-7B-Instruct
