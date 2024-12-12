# AdvAD
Official Repository for Paper: **AdvAD: Exploring Non-Parametric Diffusion for Imperceptible Adversarial Attacks [NeurIPS 2024]**

## Overview
<img src="./AdvAD_overview.png" width="90%"/>

## Dataset
ImageNet-compatible Dataset from https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset. We have downloaded and stored the images in *./dataset*. 

## Enviroment
Important packages: 
```bash
torch==1.12.1
torchvision==0.13.1
scikit-image==0.16.2
numpy==1.22.4
opencv_python==4.5.4.58
pillow==9.2.0
torchcam==0.3.2
```

## Evaluate Adversarial Examples
We have provided the adversarial examples against ResNet-50 crafted by AdvAD [here](https://drive.google.com/file/d/1GD33QM8wAojj5M4zMiBb9WqLolEiuGmL/view?usp=drive_link). Please download and unzip it in *./attack_results*, then run *eval_all.py* for evaluation.

## Craft Adversarial Examples
For AdvAD, please run:
```bash
CUDA_VISIBLE_DEVICES=0 python main_AdvAD.py
```

For AdvAD-X, please run:
```bash
CUDA_VISIBLE_DEVICES=0 python main_AdvAD_X.py
```

The configurations are set in function *create_attack_argparser()* at the end of the corresponding python file.

## Citation
```bash
@inproceedings{
li2024advad,
title={AdvAD: Exploring Non-Parametric Diffusion for Imperceptible Adversarial Attacks},
author={Jin Li and Ziqiang He and Anwei Luo and Jian-Fang Hu and Z. Jane Wang and Xiangui Kang},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024}
}
```