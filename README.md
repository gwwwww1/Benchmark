<div style="text-align: justify;">

# Evaluating Multimodal Pathology Foundation Models for Clinical Readiness in Region-of-Interest Tasks Across Organs and Supervision Regimes
<div align="center">
  <p>
    <a href="#-summary-of-pathological-datasets">📊 Datasets (ours and public)</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#-Installation">🚀 Usage</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="#-evolution-of-foundation-models-2021-2025">🧬 Evaluated Models</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://gwwwww1.github.io/FairPath/">🌐 Our Website</a>
  </p>
</div>
<hr style="border: 0; border-top: 1px solid #d0d7de; margin: 20px 0; width: 80%;">
<div align="center">
  <img src="DT.png" alt="FairPath Overview" width="850">
</div>

## 📝 Abstract
The rapid emergence of foundation models for computational pathology has generated broad interest, yet it remains unclear whether these models genuinely understand tissue at the region level–a capability essential for higher-order tasks such as morphological reasoning and report generation. Pathologists interpret tissue by examining localized regions within their surrounding context, where diagnostic signals are often spatially heterogeneous. However, existing benchmarks focus on whole-slide-level tasks and in-distribution evaluation, overlooking fine-grained morphology and out-of-distribution variability. This limitation is particularly evident at the region-of-interest (ROI) level, where lesion identification and localized risk assessment remain underexplored. Here, we present a large-scale benchmark for evaluating multimodal histopathology foundation models in ROI-level pathology. We evaluate 16 vision-language models and 4 vision-only models across 14 datasets spanning multiple organs, disease types, and tissue contexts. We identify a regime-dependent tradeoff: the best-performing vision-language models outperform the best-performing vision-only models by approximately 2% under limited supervision, whereas vision-only models consistently achieve the strongest performance under full supervision. Zero-shot performance is sensitive to prompt design, with gains from multi-template averaging and standardized labels. Standard preprocessing strategies, including image augmentation and stain normalization, yield limited improvements, with most gains below 5%. Because limited public pediatric data increases reliance on adult-centric models with uncertain applicability to pediatric populations, we curate and release five pathologist-annotated datasets of pediatric low-grade glioma subtypes under a privacy-preserving evaluation setting. These datasets establish a fine-grained ROI-level benchmark and enable systematic evaluation in pediatric pathology. Together, this work provides an evaluation framework for multimodal pathology foundation models at clinically relevant resolution and offers a measured basis for assessing where current models stand and where they fall short.

---

## 📂 Project Structure
```text
./
├── benchmarks/                 # Core benchmarking modules and evaluation logic
│   ├── clip_benchmark/         # Core benchmarking modules
│   │   ├── datasets/           # Dataset encapsulation for DataLoader
│   │   ├── histaug/            # Image enhancement & chromosome normalization
│   │   ├── metrics/            # Evaluation metrics calculation
│   │   ├── models/             # Checkpoints and configuration files
│   │   └── cli.py              # Main execution entry
│   ├── models.txt              # Foundation models registry
│   ├── models_vl.txt           # Vision-language models registry
│   ├── features/               # Extracted feature storage
│   ├── few_shot_classification.sh
│   ├── few_shot_classification_aug.sh
│   ├── image_retrieval.sh
│   ├── image_text_retrieval.sh
│   ├── linear_probe.sh
│   ├── linear_probe_aug.sh
│   └── zero_shot_classification.sh
└── dataset/                    # Raw/Processed dataset storage
```

---

## 🛠 Installation
```bash
# Clone the repository
git clone <repository-url>
cd benchmark
# Python: Install environment
conda env create -f environment.yml
```

## 🚀 Usage
All experiment scripts are stored under the benchmarks/ folder. Below are running examples for different tasks.
###  Zero-Shot Classification
```bash
cd benchmarks
bash ./zero_shot_classification.sh
```
### Few-Shot Classification
```bash
cd benchmarks
bash ./few_shot_classification.sh
# augmentation experiments
bash ./few_shot_classification.sh
```
###  Full-Sample Classification
```bash
cd benchmarks
bash  ./linear_probe.sh
# augmentation experiments
bash ./linear_probe_aug.sh
```
###  Image-only retrieval
```bash
cd benchmarks
bash ./image_retrieval.sh
```
###  Cross-modal image-text retrieval
```bash
cd benchmarks
bash  ./image_text_retrieval.sh
```
For more details, see [full documentation](./README_command.md).

## 📊 Summary of Pathological Datasets
The following table summarizes the publicly accessible datasets used in this study, encompassing various cancer types and pathological scenes.
### Our datasets
<div align="center">
  <img src="new_dataset.png" alt="FairPath Overview" width="850">
</div>

| Dataset Name | Tumor Type / Domain | Resolution | Samples | Classes | Source Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PA (Ours)** | Pilocytic Astrocytoma | 224×224 | 7,812 | 8 | [Link](https://zenodo.org/records/18961361) |
| **PMA (Ours)** | Pilomyxoid Astrocytoma | 224×224 | 16,512 | 9 | [Link](https://zenodo.org/records/18961361) |
| **DIA (Ours)** | Desmoplastic Infantile Astrocytoma | 224×224 | 20,813 | 5 | [Link](https://zenodo.org/records/18961361) |
| **PXA (Ours)** | Pleomorphic Xanthoastrocytoma | 224×224 | 19,737 | 10 | [Link](https://zenodo.org/records/18961361) |
| **SGCA (Ours)** | Subependymal Giant Cell Astrocytoma | 224×224 | 42,885 | 10 | [Link](https://zenodo.org/records/18961361) |

---
### Public datasets
| Dataset Name | Tumor Type / Domain | Resolution | Samples | Classes | Source Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PCam** | Breast Cancer | 96×96 | 327,680 | 2 | [Link](https://patchcamelyon.grand-challenge.org/) |
| **NCT-CRC-HE-100K** | Colorectal Cancer | 224×224 | 100,000 | 9 | [Link](https://zenodo.org/record/1214456) |
| **NCT-CRC-100K-norm** | Colorectal Cancer | 224×224 | 100,000 | 9 | [Link](https://zenodo.org/record/1214456) |
| **CRC100K-Val** | Colorectal Cancer | 224×224 | 7,180 | 9 | [Link](https://zenodo.org/record/1214456) |
| **Osteo** | Osteosarcoma | 1024×1024 | 1,144 | 3 | [Link](https://cancerimagingarchive.net/collection/osteosarcoma-tumor-assessment/) |
| **ARCH** (Book/Pubmed) | Multiple (Vision-Language) | - | 15,164 | Captions | [Link](https://warwick.ac.uk/fac/cross_fac/tia/data/arch) |
| **PathMMU** | Multiple (Multimodal) | - | 7,774 | Captions | [Link](https://huggingface.co/datasets/jamessyx/PathMMU) |
| **SkinCancer** | Skin Cancer | 395×395 | 129,364 | 16 | [Link](https://www.isic-archive.com/) |
| **LC25000** | Lung & Colon Cancer | 768×768 | 25,000 | 5 | [Link](https://github.com/tampapath/lung_colon_image_set) |
| **PanNuke** | Multiple (Nuclei Inst.) | 256×256 | 6,243 | 3 | [Link](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) |
| **UniToPatho** | Colorectal Polyps | - | 9,536 | 6 | [Link](https://ieee-dataport.org/open-access/unitopatho) |
| **WSSS4LUAD** | Lung Adenocarcinoma | - | 10,091 | 2 | [Link](https://wsss4luad.grand-challenge.org/WSSS4LUAD/) |
| **BRACS** | Breast Cancer Subtypes | - | 4,539 | 7 | [Link](https://www.bracs.icar.cnr.it/) |
| **SICAPv2** | Prostate Cancer | - | 18,783 | 4 | [Link](https://data.mendeley.com/datasets/9xxm58dvs3/1) |
| **GCHTID** | Gastric Cancer | 224×224 | 31,096 | 8 | [Link](https://figshare.com/articles/dataset/Gastric_Cancer_Histopathology_Tissue_Image_Dataset_GCHTID_/24087768) |

---

## 🧬 Evolution of Foundation Models (2021-2025)
This table summarizes the mainstream biomedical and pathology foundation models, sorted by release date, with direct hyperlinks to their official repositories or model weights.

| Date | Model Name | Backbone | Parameters | Size (GB) |
| :--- | :--- | :--- | :--- | :--- |
| 2021-01 | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | ViT-B/32 | ~151.28M | 5.64 |
| 2021-01 | [CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16) | ViT-B/16 | ~149.62M | 5.62 |
| 2021-12 | [PubMedCLIP ResNet-50](https://huggingface.co/sarahESL/PubMedCLIP) | ResNet-50 | ~102.01M | 5.00 |
| 2021-12 | [PubMedCLIP ResNet-50×4](https://huggingface.co/sarahESL/PubMedCLIP) | ResNet-50×4 | ~178.30M | 5.99 |
| 2021-12 | [PubMedCLIP ViT-B/32](https://huggingface.co/sarahESL/PubMedCLIP) | ViT-B/32 | ~151.28M | 5.64 |
| 2023-03 | [PMC-CLIP](https://huggingface.co/datasets/axiong/pmc_oa) | ViT-L/14 | ~427.62M | 9.24 |
| 2023-06 | [QuiltNet ViT-B/32](https://huggingface.co/wisdomik/QuiltNet-B-32) | ViT-B/32 | ~151.28M | 5.64 |
| 2023-06 | [QuiltNet ViT-B/16](https://huggingface.co/wisdomik/QuiltNet-B-16) | ViT-B/16 | ~149.62M | 5.62 |
| 2023-06 | [QuiltNet ViT-B/16-PMB](https://huggingface.co/wisdomik/QuiltNet-B-16-PMB) | ViT-B/16 + PubMedBERT | ~195.90M | 6.22 |
| 2023-07 | [CONCH](https://huggingface.co/MahmoodLab/CONCH) | ViT-B/16 | ~395.23M | 8.82 |
| 2023-08 | [PLIP](https://huggingface.co/vinid/plip) | ViT-B/32 | ~151.28M | 5.64 |
| 2023-08 | [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | ViT-B/16 + PubMedBERT | ~195.90M | 6.22 |
| 2023-08 | [PathCLIP](https://huggingface.co/jamessyx/pathclip) | ViT-B/16 | ~149.62M | 5.62 |
| 2023-08 | [UNI](https://huggingface.co/MahmoodLab/UNI) | ViT-L/14 | ~303.35M | 7.62 |
| 2024-05 | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | ViT-G/14 | ~1.1B | 18.00 |
| 2024-06 | [PathGenCLIP](https://huggingface.co/jamessyx/PathGen-CLIP) | ViT-B/16 | ~149.62M | 5.62 |
| 2024-06 | [PathGenCLIP-L](https://huggingface.co/jamessyx/PathGen-CLIP-L) | ViT-L/14 | ~427.62M | 9.24 |
| 2024-08 | [Virchow2](https://huggingface.co/paige-ai/Virchow2) | ViT-H/14 | ~631.24M | 11.89 |
| 2025-01 | [MUSK](https://github.com/lilab-stanford/MUSK) | ViT-L/16 | ~675.19M | 12.47 |
| 2025-09 | [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) | ViT-H/14 | ~681.39M | 12.55 |






