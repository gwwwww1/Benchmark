#!/bin/bash

python3 -m clip_benchmark.cli eval --pretrained_model models.txt \
        --dataset "BRACS_Rol" "BRACS_Rol_seven" \
        --task "image_retrieval" \
        --batch_size 256 \
        --num_workers 8 \
        --seed 41 \
        --dataset_root "../dataset" \
        --corrupt 0 \
        --corrupt_level "severe" \
        --corrupt_all 0 \
        --corruption_types "contrast" \
        --output "./results/zeroshot_image_retrieval/result.json"