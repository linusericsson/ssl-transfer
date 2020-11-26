# How Well Do Self-Supervised Models Transfer?
This repository hosts the code for the experiments in the paper ("How Well Do Self-Supervised Models Transfer?")[].

## Requirements
This codebase has been tested with the following package versions:
```
python=3.6.8
torch=1.2.0
torchvision=0.4.0
PIL=7.1.2
numpy=1.18.1
pandas=1.0.3
tqdm=4.31.1
sklearn=0.22.2
```

## Pre-trained models
In the paper we evaluate 14 pre-trained ResNet50 models, 13 self-supervised and 1 supervised.
To download and prepare all models in the same format, run:
```
python download_and_prepare_models.py
```
This will prepare the models in the same format and save them in a directory named `models`.

**Note**: For SimCLR-v1 and SimCLR-v2, the TensorFlow checkpoints need to be downloaded manually (using the links in the table below) and converted into PyTorch format (using https://github.com/tonylins/simclr-converter and https://github.com/Separius/SimCLRv2-Pytorch, respectively).

Below are links to the pre-trained weights used.

| Model | URL |
|-------|-----|
| InsDis | https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACcsSIt1_Njv7GsmsuzZ6Sta/InsDis.pth |
| MoCo-v1 | https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar |
| PCL-v1 | https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v1_epoch200.pth.tar |
| PIRL | https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth |
| PCL-v2 | https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v2_epoch200.pth.tar |
| SimCLR-v1 | https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_1x.zip |
| MoCo-v2 | https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar |
| SimCLR-v2 | https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/pretrained/r50_1x_sk0 |
| SeLa-v2 | https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar |
| InfoMin | https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAAzMTynP3Qc8mIE4XWkgILUa/InfoMin_800.pth |
| BYOL | https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl |
| DeepCluster-v2 | https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar |
| SwAV | https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar |
| Supervised | We use weights from `torchvision.models.resnet50(pretrained=True)` |

## Datasets
There are several classes defined in the `datasets` directory. The data is expected in a directory name `data`, located on the same level as this repository. Below is an outline of the expected file structure:
```
data/
    CIFAR10/
    DTD/
    ...
ssl-transfer/
    datasets/
    models/
    readme.md
    ...
```

## Many-shot evaluation
We provide the code for our many-shot evaluation in `many_shot_eval.py`.

To evaluate DeepCluster-v2 on CIFAR10 given our pre-computed best regularisation hyperparameter, run:
```
python many_shot_eval.py --dataset cifar10 --model deepcluster-v2 --C 0.1
```
The test accuracy should be close to 94.07%, the value reported in Table 1 of the paper.

To evaluate the Supervised baseline, run:
```
python many_shot_eval.py --dataset cifar10 --model supervised --C 0.056
```
This model should achieve close to 91.47%.

To search for the best regularisation hyperparameter on the validation set, exclude the `--C` argument:
```
python many_shot_eval.py --dataset cifar10 --model supervised
```

## Few-shot evaluation
We provide the code for our few-shot evaluation in `few_shot_eval.py`.

To evaluate DeepCluster-v2 on EuroSAT in a 5-way 5-shot setup, run:
```
python few_shot_eval.py --dataset eurosat --model deepcluster-v2 --n-way 5 --n-support 5
```
The test accuracy should be close to 88.39% ± 0.49%, the value reported in Table 2 of the paper.

Or, to evaluate the Supervised baseline on ChestX in a 5-way 50-shot setup, run:
```
python few_shot_eval.py --dataset chestx --model supervised --n-way 5 --n-support 50
```
This model should achieve close to 32.34% ± 0.45%.
