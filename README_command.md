# Benchmark Evaluation Scripts Reference
This guide covers all evaluation scripts used for the project, categorized by task type.

---

## 1. Linear Probe Benchmarking
These scripts assess the classification performance of models using linear probing under various supervision regimes (Few-shot and Full-supervision).

### Few-Shot Evaluation
```bash
# 3/5/10-shot evaluation (With rotation and flipping augmentation)
python3 -m clip_benchmark.cli eval \
    --pretrained_model models.txt \               # model configure file
    --dataset "PX" "SGCA" \                       # dataset_name
    --task "linear_probe" \                       # task type
    --batch_size 256 \                      
    --num_workers 8 \
    --fewshot_k $k_shot \                         # 3, 5, or 10
    --seed $seed \                             
    --augmentation 1 \                            # Enable augmentation 1: Enable 0: Disable (default)
    --augmentation_types "rotation_flipping" \    # rotation_flipping" | "all augmentations" | "stain normalisation" 
    --dataset_root "../dataset" \                 # dataset root
    --output "./results/linear_probe/benchmark_fs_${k_shot}shot_seed${seed}.json" #result root
```
### Full-Supervision Evaluation
```bash
 #Evaluate using the full training set (k = -1)
python3 -m clip_benchmark.cli eval \
    --pretrained_model models.txt \
    --dataset "PX" "SGCA" \
    --task "linear_probe" \
    --fewshot_k -1 \                        # -1 signifies full dataset usage
    --seed $seed \
    --augmentation 1 \                      # With/Without augmentation variants
    --augmentation_types "rotation_flipping" \
    --dataset_root "../dataset" \
    --output "./results/linear_probe/benchmark_fs_full_seed${seed}.json"
```
## 2.Zero-Shot Classification
Evaluates the zero-shot capabilities of vision-language models.
```bash
#Zero-shot evaluation (Comparing Multi-template vs. Single-template)
python3 -m clip_benchmark.cli eval \
        --pretrained_model models_vl.txt \
        --dataset "PX" "PXA" "DIA" "SGCA" \
        --task "zeroshot_classification" \
        --batch_size 256 \
        --seed 41 \
        --corrupt 0 \                       # Disable image corruption testing
        --corrupt_level "severe" \
        --single_template $flag \           # 0 for Multi-template, 1 for Single
        --output "./results/zeroshot_classification/result_${flag}.json"
```
## 3. Retrieval Tasks
Evaluates cross-modal alignment or image-to-image similarity.
### Cross-modal Retrieval
```bash
python3 -m clip_benchmark.cli eval \
      --pretrained_model models_vl.txt \
      --dataset "pubmedset_retrieval" "pathmmu_retrieval" "bookset_retrieval" \
      --task "zeroshot_retrieval" \
      --recall_k 1 10 50 \                  # Recall at K metrics
      --dataset_root "../dataset" \
      --output "./results/zeroshot_retrieval/result.json"
```
### Image Retrieval
```bash
python3 -m clip_benchmark.cli eval \
        --pretrained_model models.txt \
        --dataset "BRACS_Rol" "BRACS_Rol_seven" \
        --task "image_retrieval" \
        --batch_size 256 \
        --seed 41 \
        --corrupt 0 \                       # Disable corruption
        --corruption_types "contrast" \     # Specific corruption type
        --dataset_root "../dataset" \
        --output "./results/zeroshot_image_retrieval/result.json"
```





