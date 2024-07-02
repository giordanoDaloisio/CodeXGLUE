#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -J cloze
#SBATCH -o ./cloze_%j.out
#SBATCH -p cuda
#SBATCH -c 32
#SBATCH --gres=gpu:large

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd code

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codecompl

srun python run_cloze.py \
			--model microsoft/codebert-base-mlm \
			--cloze_mode all \
			--lang ruby \
			--output_dir evaluator/predictions/ \
			--no_cuda