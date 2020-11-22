# How Well Do Self-Supervised Models Transfer?
This repository hosts the code for the experiments in the CVPR 2021 submission.

## Requirements
This codebase has been tested with the following package versions:
```
python=3.6.8
torch=1.2.0
torchvision=0.4.0
PIL=7.1.2
numpy=1.18.1
tqdm=4.31.1
sklearn=0.22.2
```

## Pre-trained models

| Model | URL |
|-------|-----|
| supervised | link |
| simclr-v1 | link |
| simclr-v2 | link |
| moco-v1 | https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar |
| moco-v2 | https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar |
| byol | https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl |
| swav | https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar |
| deepcluster-v2 | https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar |
| sela-v2 | https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar |
| infomin | https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAAzMTynP3Qc8mIE4XWkgILUa/InfoMin_800.pth |
| insdis | https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACcsSIt1_Njv7GsmsuzZ6Sta/InsDis.pth |
| pirl | https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth |
| pcl-v1 | https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v1_epoch200.pth.tar |
| pcl-v2 | https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v2_epoch200.pth.tar |

## Many-shot evaluation
We provide the bare essentials of our evaluation in `essential_eval.py`.

To evaluate DeepCluster-v2 on CIFAR10 given our pre-computed best regularisation hyperparameter run:
```
python essential_eval.py --dataset cifar10 --model deepcluster-v2 --C 0.562
```
The test accuracy should be close to 94.07%, the value reported in Table 1 of the paper.

To evaluate the Supervised baseline, run:
```
python essential_eval.py --dataset cifar10 --model supervised --C 0.056
```
This model should achieve close to  91.47%.
