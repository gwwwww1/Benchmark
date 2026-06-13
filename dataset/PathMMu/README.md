---
license: cc-by-nd-4.0
extra_ated_heading:  "Access PathMMU on Hugging Face" 
extra_gated_prompt: "This is a form to enable access to PathMMU on Hugging Face after you have been granted access. Please fill the [**Google form**](https://docs.google.com/forms/d/e/1FAIpQLSfbWXatC9j2bQ-X0NULngZf5ydqrBxV3UgrM-So4cypOIH9bQ/viewform?vc=0&c=0&w=1&flr=0) and accept our license terms and acceptable use policy before submitting this form. Your Hugging Face account email address MUST match the email you provide on the Google form, or your request will not be approved. Requests will be processed in 1-2 days."
extra_gated_fields:
  Country: country
  Specific date: date_picker
  I want to use this dataset for:
    type: select
    options: 
      - Research
      - Education
      - label: Other
        value: other
  I agree to use this dataset for non-commercial use ONLY: checkbox
---
This is the official Hugging Face repo for [**PathMMU: A Massive Multimodal Expert-Level Benchmark for Understanding and Reasoning in Pathology**](https://arxiv.org/abs/2401.16355) 

[**🌐 Homepage**](https://pathmmu-benchmark.github.io/) | [**🤗 Dataset**](https://huggingface.co/datasets/jamessyx/PathMMU/) | [**📖 arXiv**](https://arxiv.org/abs/2401.16355) | [**GitHub**](https://github.com/PathMMU-Benchmark/PathMMU)


## 🔔News
- **Important Notes!!!!!!** 
- The benchmark data and evaluation code have been released (8/7/2024)

## Abstract
The emergence of large multimodal models has unlocked remarkable potential in AI, particularly in pathology. However, the lack of specialized, high-quality benchmark impeded their development and precise evaluation. To address this, we introduce PathMMU, the largest and highest-quality expert-validated pathology benchmark for Large Multimodal Models (LMMs). It comprises 33,428 multimodal multi-choice questions and 24,067 images from various sources, each accompanied by an explanation for the correct answer. The construction of PathMMU harnesses GPT-4V's advanced capabilities,  utilizing over 30,000 image-caption pairs to enrich captions and generate corresponding Q\&As in a cascading process. Significantly, to maximize PathMMU's authority, we invite seven pathologists to scrutinize each question under strict standards in PathMMU's validation and test sets, while simultaneously setting an expert-level performance benchmark for PathMMU. We conduct extensive evaluations, including zero-shot assessments of 14 open-sourced and 4 closed-sourced LMMs and their robustness to image corruption. We also fine-tune representative LMMs to assess their adaptability to PathMMU. The empirical findings indicate that advanced LMMs struggle with the challenging PathMMU benchmark, with the top-performing LMM, GPT-4V, achieving only a 49.8\% zero-shot performance, significantly lower than the 71.8\% demonstrated by human pathologists. After fine-tuning, significantly smaller open-sourced LMMs can outperform GPT-4V but still fall short of the expertise shown by pathologists. We hope that the PathMMU will offer valuable insights and foster the development of more specialized, next-generation LMMs for pathology.

## Citation

```
@misc{sun2024pathmmu,
      title={PathMMU: A Massive Multimodal Expert-Level Benchmark for Understanding and Reasoning in Pathology}, 
      author={Yuxuan Sun and Hao Wu and Chenglu Zhu and Sunyi Zheng and Qizi Chen and Kai Zhang and Yunlong Zhang and Xiaoxiao Lan and Mengyue Zheng and Jingxiong Li and Xinheng Lyu and Tao Lin and Lin Yang},
      year={2024},
      eprint={2401.16355},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Data Acquisition

All QAs and PubMed images, as well as EduContent images, can be accessed directly on Huggingface. For images from other subsets, please follow the [instructions](https://github.com/PathMMU-Benchmark/PathMMU/blob/main/data/instructions.md) to obtain them.
Please place the downloaded data into the `data` folder.

## Acknowledgement

Our project utilizes part of the raw data sourced from [**OpenPath**](https://www.nature.com/articles/s41591-023-02504-3) and [**Quilt-1M**](https://arxiv.org/abs/2306.11207). We extend our gratitude for their significant contributions to the community. 

**Quilt-1M:**

```
@article{ikezogwo2023quilt,
  title={Quilt-1M: One Million Image-Text Pairs for Histopathology},
  author={Ikezogwo, Wisdom Oluchi and Seyfioglu, Mehmet Saygin and Ghezloo, Fatemeh and Geva, Dylan Stefan Chan and Mohammed, Fatwir Sheikh and Anand, Pavan Kumar and Krishna, Ranjay and Shapiro, Linda},
  journal={arXiv preprint arXiv:2306.11207},
  year={2023}
}
```

**OpenPath:**

```
@article{huang2023visual,
  title={A visual--language foundation model for pathology image analysis using medical twitter},
  author={Huang, Zhi and Bianchi, Federico and Yuksekgonul, Mert and Montine, Thomas J and Zou, James},
  journal={Nature medicine},
  volume={29},
  number={9},
  pages={2307--2316},
  year={2023},
  publisher={Nature Publishing Group US New York}
}
```