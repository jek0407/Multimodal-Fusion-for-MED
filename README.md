### This project will perform Multimedia Event Detection(MED) with Multi-modal Fusion.

The final features to be used is a fusion of SoundNet features & 3D CNN features.
In this project, we will try various fusion methods using pre-extracted features (snf, cnn3d) to extract new features.
Finally, the fusion features will be used to train and test the mlp, the classifier.

## Recommended Hardware

This code template is built based on [PyTorch](https://pytorch.org) and [Pyturbo](https://github.com/CMU-INF-DIVA/pyturbo) for Linux to fully utilize the computation of multiple CPU cores and GPUs.
SIFT feature, K-Means, and Bag-of-Words must run on CPUs, while CNN features and MLP classifiers can run on GPUs.
For GCP, an instance with 16 vCPU (e.g. `n1-standard-16`) and a Nvidia T4 GPU instance should be sufficient for the full pipeline.
During initial debugging, you are recommended to use a smaller instance to save money, e.g., `n1-standard-1` (only 1 vCPU) with Nvidia T4 or without GPU for the SIFT part.

## Install Dependencies

A set of dependencies is listed in [environment.yml](environment.yml). You can use `conda` to create and activate the environment easily.

```bash
# Start from within this repo
conda env create -f environment.yml -p ./env
conda activate ./env
```

## Check CUDA version

```bash
nvidia-smi
```
Check CUDA version of your device. And install the right version of pytorch(torchvision)[PyTorch](https://pytorch.org/get-started/previous-versions/)

## Dataset

This project uses pre-extracted features from here : [DATA](https://github.com/KevinQian97/11755-ISR-HW1#data-and-labels) 
Therefore, there is no need to download the data.


## Directory structures

* this repo
  * code
  * data
    * cnn3d (features pre-extracted https://github.com/jek0407/Video-based-MED)
    * labels
    * snf (features pre-extracted https://github.com/jek0407/SoundNet)
  * env
  * ...

## About Features

We had extracted features in advance, 

* snf (SonudNet Features)
  * Check this repository : [SoudNet](https://github.com/jek0407/SoundNet)
* cnn3d (3D CNN Features)
  * Check this repository : [Video](https://github.com/jek0407/Video-based-MED)

Through various fusion methods, we extract new features that are expected to be better.


## MLP Classifier

The training script automatically and deterministically split the `train_val` data into training and validation, so you do not need to worry about it.

### Uni-Modal

To train MLP with SoundNet features, run

```bash
python code/run_mlp.py snf --feature_dir data/snf --num_features 255
```

By default, training logs and predictions are stored under `data/mlp/snf/version_xxx/`. (Kaggle score : 0.533)


To train MLP with 3D CNN features, run

```bash
python code/run_mlp.py cnn3d --feature_dir data/cnn3d --num_features 512
```

By default, training logs and predictions are stored under `data/mlp/cnn3d/version_xxx/`. (Kaggle score : 0.9439)


### Multi-Modal
 
1) To train with Multimodal Early Fusion features, run
```bash
python code/run_early_fusion_mlp.py ealry_fusion --feature_dir1 data/cnn3d --feature_dir2 data/snf --num_features 767
```
num_features = 255(snf) + 512(cnn3d)

By default, training logs and predictions are stored under `data/mlp/early_fusion/version_xxx/`. (Kaggle score : 0.9599)

2) To train with Multimodal Late Fusion features, run
```bash
python code/run_late_fusion_mlp.py late_fusion 
```
'num_features' and 'feature_dir' are defined in the code.

By default, training logs and predictions are stored under `data/mlp/late_fusion/version_xxx/`. (Kaggle score : 0.???)

2) To train with Multimodal Double Fusion features, run
```bash
python code/run_double_fusion_mlp.py double_fusion 
```
'num_features' and 'feature_dir' are defined in the code.

By default, training logs and predictions are stored under `data/mlp/double_fusion/version_xxx/`. (Kaggle score : 0.???)

### This project was from CMU 11-775 Fall 2023 Homework 2
See [PDF Handout](docs/handout.pdf)