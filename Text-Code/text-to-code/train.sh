#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs/train_%j.out
#SBATCH -J code_gen_train
#SBATCH -p cuda
#SBATCH -c 40
#SBATCH --gres=gpu:large

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

cd code

LANG=java
DATADIR=../dataset/concode
OUTPUTDIR=../save/concode
PRETRAINDIR=microsoft/CodeGPT-small-java-adaptedGPT2    # will download pre-trained CodeGPT model
LOGFILE=text2code_concode.log

python run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=512 \
        --do_train \
        --node_index 0 \
        --learning_rate=5e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=6 \
        --per_gpu_eval_batch_size=12 \
        --gradient_accumulation_steps=2 \
        --num_train_epochs=30 \
        --logging_steps=100 \
        --save_steps=5000 \
        --overwrite_output_dir \
        --seed=42