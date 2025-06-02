#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs_t5/train_%j.out
#SBATCH -J defect
#SBATCH -p cuda
#SBATCH -c 40
#SBATCH --gres=gpu:fat

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

base_model=Salesforce/codet5p-770m
model_type=t5
output_dir=./saved_models_t5

cd code
srun python run_lora_simple.py --model_name $base_model --output_dir $output_dir

# python run.py \
#     --output_dir=./saved_models_t5 \
#     --model_type=t5 \
#     --tokenizer_name=Salesforce/codet5p-770m \
#     --model_name_or_path=Salesforce/codet5p-770m \
#     --do_train \
#     --train_data_file=../dataset/train.jsonl \
#     --eval_data_file=../dataset/valid.jsonl \
#     --test_data_file=../dataset/test.jsonl \
#     --epoch 5 \
#     --block_size 500 \
#     --train_batch_size 32 \
#     --eval_batch_size 64 \
#     --learning_rate 2e-5 \
#     --max_grad_norm 1.0 \
#     --evaluate_during_training \
#     --seed 123456  2>&1 | tee train.log
