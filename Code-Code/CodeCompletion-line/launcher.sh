#!/bin/bash -l
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./codecompl.out
#SBATCH -J codecompl
#SBATCH -p cuda
#SBATCH -c 8
#SBATCH --gres=gpu:large

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codecompl

export CUDA_VISIBLE_DEVICES=0
cd ./code
LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/line_completion
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=../../CodeCompletion-token/save/javaCorpus/checkpoint
LOGFILE=completion_javaCorpus_eval.log


srun python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --eval_line \
        --logging_steps=100 \
        --seed=42 



srun python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs
