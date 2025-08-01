#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs_llama/test_%j.out
#SBATCH -J def_cuda
#SBATCH -p normal
#SBATCH -c 60

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

base_model=meta-llama/Llama-3.1-8B
model_path=./saved_models_llama/final_model
model_type=llama
output_dir=./saved_models_llama

cd code
srun python run.py \
    --output_dir=$output_dir \
    --model_type=$model_type \
    --tokenizer_name=$base_model \
    --model_name_or_path=$model_path \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --job_id $SLURM_JOB_ID \
    --seed 123456 2>&1 \
    "$@" | tee test.log