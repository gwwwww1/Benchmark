#!/bin/bash



# 定义 shot 列表
shot_list=(1)  # 这里可以根据需求修改 shot 的值
# 定义随机种子列表123
seed_list=(123 234 345 456 567 678 789 890 901 012)  # 这里可以根据需求修改随机种子的值
#234 345 456 567 678 789 890 901 012
# 循环遍历 shot 列表
for k_shot in "${shot_list[@]}"
do
  # 循环遍历随机种子列表
  for seed in "${seed_list[@]}"
  do
      # 执行评估命令
      python3 -m clip_benchmark.cli eval --pretrained_model models2.txt \
          --dataset "SGCA" \
          --task "linear_probe" \
          --batch_size 256 \
          --num_workers 8 \
          --fewshot_k $k_shot \
          --seed $seed \
          --dataset_root "../dataset" \
          --output "./results/linear_probe/benchmark_fs_${k_shot}shot_seed${seed}.json"
  done
done
