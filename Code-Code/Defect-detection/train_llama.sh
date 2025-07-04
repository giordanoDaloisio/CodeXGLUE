#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs_llora/train_%j.out
#SBATCH -J defect
#SBATCH -p cuda
#SBATCH -c 40
#SBATCH --gres=gpu:fat

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

base_model=unsloth/Meta-Llama-3.1-8B
model_type=llama
output_dir=./saved_models_llama

cd code
python run_lora_simple.py --model_name $base_model --output_dir $output_dir
