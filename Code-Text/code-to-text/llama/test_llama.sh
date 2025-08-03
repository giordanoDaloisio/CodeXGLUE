#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs_llama/test_cuda_%j.out
#SBATCH -J llama_test
#SBATCH -p cuda
#SBATCH -c 10
#SBATCH --gres=gpu:3c_s80g:1
# SBATCH --gres=gpu:fat


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd code
lang=java #programming language
lr=5e-5
batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=model_llama/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl

epochs=10
pretrained_model=meta-llama/Llama-3.1-8B
model_type=t5

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

python run_llama_lora.py \
    --model_name_or_path $pretrained_model \
    --output_dir $output_dir \
    --train_filename $train_file \
    --dev_filename $dev_file \
    --test_filename $test_file \
    --do_test \
    --lora_r 16 \
    --lora_alpha 32 \
    --learning_rate 2e-4 \
    --train_batch_size 4 \
    --num_train_epochs 2 \
    --job_id $SLURM_JOB_ID \
