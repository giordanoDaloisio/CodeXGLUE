#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs_t5/save.out
#SBATCH -J defect
#SBATCH -p normal
#SBATCH -c 40


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

cd code
srun python save_model.py