#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs/test_cuda_quant_%j.out
#SBATCH -J ct_quant_cuda
#SBATCH -p cuda
#SBATCH -c 40
#SBATCH --gres=gpu:large


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd code
lang=java #programming language
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
epochs=10 
file_len=300
pretrained_model=microsoft/codebert-base #Codebert: microsoft/codebert-base #Roberta: roberta-base

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

batch_size=64
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
model=roberta

srun python run.py --do_test --model_type $model --model_name_or_path $pretrained_model --load_model_path $test_model --test_filename $test_file --file_len $file_len --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size --job_id $SLURM_JOB_ID --quantize
