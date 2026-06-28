<div align="center">

# OpenDPR: Open-Vocabulary Change Detection via Vision-Centric Diffusion-Guided Prototype Retrieval for Remote Sensing Imagery

**CVPR 2026**

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2603.27645)
[![CVPR](https://img.shields.io/badge/Paper-CVPR%20OpenAccess-007acc)](https://openaccess.thecvf.com/content/CVPR2026/html/Guo_OpenDPR_Open-Vocabulary_Change_Detection_via_Vision-Centric_Diffusion-Guided_Prototype_Retrieval_for_CVPR_2026_paper.html)
[![Code](https://img.shields.io/badge/Code-OpenDPR-181717?logo=github)](https://github.com/guoqi2002/OpenDPR)
![Task](https://img.shields.io/badge/Task-Open--Vocabulary%20Change%20Detection-green)

</div>

---

## 📢 News

- `2026/06/26`: Code and data are publicly available.
- `2026/03/31`: The paper is available on [arXiv](https://arxiv.org/abs/2603.27645).
- `2026/02/21`: OpenDPR is accepted to **CVPR 2026**.

---

## ✨ Overview

Open-vocabulary change detection (OVCD) seeks to recognize arbitrary changes of interest by enabling generalization beyond a fixed set of predefined classes. We reformulate OVCD as a two-stage pipeline: first generate class-agnostic change proposals using visual foundation models (VFMs) such as SAM and DINOv2, and then perform category identification with vision-language models (VLMs) such as CLIP. We reveal that category identification errors are the primary bottleneck of OVCD, mainly due to the limited ability of VLMs based on image-text matching to represent fine-grained land-cover categories. To address this, we propose OpenDPR, a training-free vision-centric diffusion-guided prototype retrieval framework. OpenDPR leverages diffusion models to construct diverse prototypes for target categories offline, and to perform similarity retrieval with change proposals in the visual space during inference. The secondary bottleneck lies in change localization, due to the inherent lack of change priors in VFMs. To bridge this gap, we design a spatial-to-change weakly supervised change detection module named S2C to adapt their strong spatial modeling capabilities for change localization. Integrating the pretrained S2C into OpenDPR leads to an optional weakly supervised variant named OpenDPR-W, which further improves OVCD with minimal supervision. Experimental results on four benchmark datasets demonstrate that the proposed methods achieve state-of-the-art performance under both supervision modes.

---

## 🧭 Project Structure

```text
OpenDPR/
├── dynamic_earth/              # Core model loading, proposal matching, and prototype retrieval
├── infer_OpenDPR/              # Training-free OpenDPR inference scripts
│   ├── WHU-CD/
│   ├── LEVIR-CD/
│   ├── SECOND/
│   └── Hi-UCD-mini/
├── infer_OpenDPR_W/            # OpenDPR-W inference scripts with S2C change-location priors
├── eval/                       # Evaluation scripts
├── support_set/                # Prototype support images and masks
├── S2C_results/                # Provided S2C inference results for OpenDPR-W
└── third_party/segment_anything/
```

---

## 🛠️ Installation

The released inference and evaluation code has been tested with **Python 3.9**, **PyTorch 2.1.0**, **CUDA 11.8**, **SAM**, and **DINOv2**.

```bash
conda create -n opendpr python=3.9 -y
conda activate opendpr
```

Install PyTorch according to your CUDA version. For CUDA 11.8:

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.32.1 numpy==1.26.3 opencv-python==4.8.0.76 pillow tqdm scipy scikit-image scikit-learn
```

Install Segment Anything:

```bash
cd third_party/segment_anything
pip install -e .
cd ../..
```

---

## 🗂️ Preparation

### Evaluation Datasets

Experiments are conducted on four datasets:

| Dataset | Task type | Placeholder |
| --- | --- | --- |
| [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html) | Building change detection | `<WHU_CD_ROOT>` |
| [LEVIR-CD](https://justchenhao.github.io/LEVIR/) | Building change detection | `<LEVIR_CD_ROOT>` |
| [SECOND](https://captain-whu.github.io/SCD/) | Semantic change detection | `<SECOND_ROOT>` |
| [Hi-UCD mini](https://github.com/Daisy-7/Hi-UCD-S) | Semantic change detection | `<HIUCD_ROOT>` |

Update the dataset placeholders in the corresponding inference and evaluation scripts before running.

For OpenDPR, inference is performed in a training-free manner using only the test set. For OpenDPR-W, weakly supervised pre-training is first conducted on the training set, and the model achieving the best F1 score on the validation set is used for binary change localization on the test set. Notably, as each image pair in the manually curated SECOND dataset inherently contains change, only OpenDPR is evaluated on this dataset. More details are provided in the Supplementary Material.

### Model Weights

Download the required checkpoints yourself and replace the placeholders in the scripts.

| Placeholder | Meaning | Where used |
| --- | --- | --- |
| `<SAM_CHECKPOINT>` | Path to the SAM checkpoint, such as `sam_vit_h_4b8939.pth` | `infer_OpenDPR/**/infer_*.py`, `infer_OpenDPR_W/**/infer_*.py` |
| `<DINOV2_MODEL_DIR>` | Local directory of the downloaded DINOv2 model | `dynamic_earth/utils/model.py` |
| `<DINO_V1_CHECKPOINT>` | Optional DINOv1 checkpoint path, only needed if you switch the model type to DINOv1 | `dynamic_earth/utils/model.py` |

The released inference scripts use **SAM (ViT-H)** and **DINOv2 (ViT-B/14)** by default. Users are encouraged to flexibly try other model variants.

### Implementation Details

In this work, we use [GPT-4](https://openai.com/index/gpt-4-research/) to generate diverse descriptions, [DiffusionSat](https://www.samarkhanna.com/DiffusionSat/) for image generation, and [APE](https://github.com/shenyunhang/APE) for class-specific localization. **Our method does not rely on a specific foundation model**, and users are encouraged to flexibly try stronger alternatives.

We provide the support sets used in our experiments in `support_set`. If you use the provided data, the accuracy should be close to the reported results. The performance of OpenDPR is highly dependent on the quality of the support set. Therefore, using stronger foundation models or conducting more careful support-set verification is expected to further improve the results.

---

## 🚀 Quick Start

Run commands from the repository root. The inference scripts for **OpenDPR** and **OpenDPR-W** are provided in `infer_OpenDPR` and `infer_OpenDPR_W`, respectively. In each dataset directory, **`infer_all.py` performs multi-class inference jointly**, while **`infer_building.py` and the other `infer_*.py` scripts perform class-wise inference**. We provide the binary S2C change-localization results used by OpenDPR-W in `S2C_results`. Below are some examples.

### 🧪 OpenDPR

```bash
python infer_OpenDPR/LEVIR-CD/infer_building.py
python infer_OpenDPR/WHU-CD/infer_building.py
python infer_OpenDPR/SECOND/infer_all.py
python infer_OpenDPR/Hi-UCD-mini/infer_all.py
```

### 🧩 OpenDPR-W

```bash
python infer_OpenDPR_W/LEVIR-CD/infer_building.py
python infer_OpenDPR_W/WHU-CD/infer_building.py
python infer_OpenDPR_W/Hi-UCD-mini/infer_all.py
```

Default outputs are saved under:

```text
output/{DATASET}/infer_{CLASS_OR_ALL}
```

## 📏 Evaluation

Evaluation scripts are provided in `eval`. Update the ground-truth placeholders before running.

```bash
python eval/evaluate_LEVIR-CD.py
python eval/evaluate_WHU-CD.py
python eval/evaluate_HiUCD-mini.py
python eval/evaluate_SECOND.py
```

Default prediction paths follow the output directories used by the inference scripts.

---

## 📝 Results

Quantitative results are reported in the paper and supplementary material. Qualitative visualizations will be added soon.

---

## 📌 Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{guo2026opendpr,
  title={OpenDPR: Open-Vocabulary Change Detection via Vision-Centric Diffusion-Guided Prototype Retrieval for Remote Sensing Imagery},
  author={Guo, Qi and Wang, Jue and Liu, Yinhe and Zhong, Yanfei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

---

## 🙏 Acknowledgement

This project is built upon [DynamicEarth](https://github.com/likyoo/DynamicEarth), the first OVCD codebase released by the Earth Vision Community. We sincerely thank the authors for their valuable contribution.

We also thank the contributors of [Segment Anything](https://github.com/facebookresearch/segment-anything), [DINOv2](https://github.com/facebookresearch/dinov2), [APE](https://github.com/shenyunhang/APE), [DiffusionSat](https://www.samarkhanna.com/DiffusionSat/), and the benchmark datasets used in this work.
