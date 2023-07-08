#!/bin/bash
accelerate launch ../src/train_sft.py \
    --model_name_or_path /home/amur/private/chatglm2-6b \
    --use_v2 \
    --do_train \
    --dataset POI_bj \
    --dataset_dir ../data \
    --finetuning_type p_tuning \
    --output_dir /home/amur/private/chatglm2-6b-sft \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --learning_rate 1e-3 \
    --num_train_epochs 20.0 
