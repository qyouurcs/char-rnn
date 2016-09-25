#!/usr/bin/bash

if [ $# -lt 1 ];then
    echo "Usage: $0 <gpuid>"
    exit
fi

gpuid=$1

CUDA_VISIBLE_DEVICES=$gpuid th train_cap_char_emb.lua -finetune_cnn_after 0 -start_from ./cv_emb/model_id.t7
