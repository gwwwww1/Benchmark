#!/bin/bash
seed_list=(123 234 345 456 567 678 789 890 901 012)
for seed in "${seed_list[@]}"
  do
      # 执行评估命令
      python3 -m clip_benchmark.cli eval --pretrained_model models.txt \
          --dataset  "PX" "SGCA" \
          --task "linear_probe" \
          --batch_size 256 \
          --num_workers 8 \
          --fewshot_k -1 \
          --seed $seed \
          --augmentation 1  \
          --augmentation_types "rotation_flipping" \
          --dataset_root "../dataset" \
          --output "./results/linear_probe/benchmark_fs_${k_shot}shot_seed${seed}.json"
  done