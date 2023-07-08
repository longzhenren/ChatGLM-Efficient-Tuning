#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/train_sft.py \
    --use_v2 \
    --do_predict \
    --dataset rd_test \
    --dataset_dir ../data \
    --model_name_or_path /home/amur/private/chatglm2-6b \
    --checkpoint_dir /home/amur/private/chatglm2-6b-poi-rd \
    --output_dir /home/amur/private/chatglm2-eval-rd-res \
    --overwrite_cache \
    --per_device_eval_batch_size 8 \
    --max_samples 50 \
    --predict_with_generate
