#!/bin/bash

python3 -m clip_benchmark.cli eval --pretrained_model models_vl.txt \
      --dataset "pubmedset_retrieval"  "pathmmu_retrieval" "bookset_retrieval" \
      --task "zeroshot_retrieval" \
      --batch_size 256 \
      --num_workers 8 \
      --seed 42 \
      --recall_k 1 10 50 \
      --dataset_root "../dataset" \
      --output "./results/zeroshot_retrieval/result.json"