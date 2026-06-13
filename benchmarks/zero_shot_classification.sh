#!/bin/bash

python3 -m clip_benchmark.cli eval --pretrained_model models_vl.txt \
        --dataset "PX" "PXA" "DIA" "SGCA" \
        --task "zeroshot_classification" \
        --batch_size 256\
        --num_workers 8 \
        --seed 41 \
        --dataset_root "../dataset" \
        --corrupt 0 \
        --corrupt_level "severe" \
        --corrupt_all 0 \
        --corruption_types "saturation" \
        --single_template 0 \
        --output "./results/zeroshot_classification/result_opt.json"
#single
python3 -m clip_benchmark.cli eval --pretrained_model models_vl.txt \
        --dataset "PX" "PXA" "DIA" "SGCA" \
        --task "zeroshot_classification" \
        --batch_size 256\
        --num_workers 8 \
        --seed 41 \
        --dataset_root "../dataset" \
        --corrupt 0 \
        --corrupt_level "severe" \
        --corrupt_all 0 \
        --corruption_types "saturation" \
        --single_template 1 \
        --output "./results/zeroshot_classification/result_single_opt.json"