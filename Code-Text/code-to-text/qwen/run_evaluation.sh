#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs_qwen/eval_%j.out
#SBATCH -J eval
#SBATCH -p cuda
#SBATCH -c 10
#SBATCH --gres=gpu:large

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

beam_size=10
target_length=128

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

srun python evaluator/evaluator_qwen.py --file code_summarization_results_None.json
