# How Well Do Self-Supervised Models Transfer?
This repository hosts the code for the experiments in the paper [How Well Do Self-Supervised Models Transfer?](https://arxiv.org/abs/2011.13377)

## Requirements
This codebase has been tested with the following package versions:
```
python=3.6.8
torch=1.2.0
torchvision=0.4.0
PIL=7.1.2
numpy=1.18.1
scipy=1.2.1
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

**Note 1**: For SimCLR-v1 and SimCLR-v2, the TensorFlow checkpoints need to be downloaded manually (using the links in the table below) and converted into PyTorch format (using https://github.com/tonylins/simclr-converter and https://github.com/Separius/SimCLRv2-Pytorch, respectively).

**Note 2**: In order to convert BYOL, you may need to install some packages by running:
```
pip install jax jaxlib dill git+https://github.com/deepmind/dm-haiku
```

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

## Many-shot linear evaluation
We provide the code for our linear evaluation in `linear.py`.

To evaluate DeepCluster-v2 on CIFAR10 given our pre-computed best regularisation hyperparameter, run:
```
python linear.py --dataset cifar10 --model deepcluster-v2 --C 0.316
```
The test accuracy should be close to 94.07%, the value reported in Table 1 of the paper.

To evaluate the Supervised baseline, run:
```
python linear.py --dataset cifar10 --model supervised --C 0.056
```
This model should achieve close to 91.47%.

To search for the best regularisation hyperparameter on the validation set, exclude the `--C` argument:
```
python linear.py --dataset cifar10 --model supervised
```

Finally, when using SimCLR-v1 or SimCLR-v2, use the --no-norm argument:
```
python linear.py --dataset cifar10 --model simclr-v1 --no-norm
```

## Many-shot finetuning
We now also provide code for finetuning in `finetune.py`.

To finetune DeepCluster-v2 on CIFAR10, run:
```
python finetune.py --dataset cifar10 --model deepcluster-v2
```

## Few-shot evaluation
We provide the code for our few-shot evaluation in `few_shot.py`.

To evaluate DeepCluster-v2 on EuroSAT in a 5-way 5-shot setup, run:
```
python few_shot.py --dataset eurosat --model deepcluster-v2 --n-way 5 --n-support 5
```
The test accuracy should be close to 88.39% ± 0.49%, the value reported in Table 2 of the paper.

Or, to evaluate the Supervised baseline on ChestX in a 5-way 50-shot setup, run:
```
python few_shot.py --dataset chestx --model supervised --n-way 5 --n-support 50
```
This model should achieve close to 32.34% ± 0.45%.

## Object detection
We use the [detectron2](https://github.com/facebookresearch/detectron2) framework to train our models on PASCAL VOC object detection.

Below is an outline of the expected file structure, including config files, converted models and the detectron2 framework:
```
detectron2/
    tools/
        train_net.py
        ...
    ...
ssl-transfer/
    detectron2-configs/
        finetune/
            byol.yaml
            ...
        frozen/
            byol.yaml
            ...
    models/
        detectron2/
            byol.pkl
            ...
        ...
    ...
```

To set it up, perform the following steps:
1. [Install detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (requries PyTorch 1.5 or newer). We expect the installed framework to be located at the same level as this repository, see outline of expected file structure above.
2. Convert the models into the format used by detectron2 by running `python convert_to_detectron2.py`. The converted models will be saved in a directory called `detectron2` inside the `models` directory.

We include the config files for the frozen training in `detectron2-configs/frozen` and for full finetuning in `detectron2-configs/finetune`.
In order to train models, navigate into `detectron2/tools/`. We can now train e.g. BYOL with a frozen backbone on 1 GPU by running:
```
./train_net.py --num-gpus 1 --config-file ../../ssl-transfer/detectron2-configs/frozen/byol.yaml OUTPUT_DIR ./output/byol-frozen
```
This model should achieve close to 82.01 AP50, the value reported in Table 3 of the paper.

## Surface Normal Estimation
The code for running the surface normal estimation experiments is given in the `surface-normal-estimation`. We use the [MIT CSAIL Semantic Segmentation Toolkit](https://github.com/CSAILVision/semantic-segmentation-pytorch), but there is also a docker configuration file that can be used to build a container with all the dependencies installed. One can train a model with a command like:

```
./scripts/train_finetune_models.sh <pretrained-model-path> <checkpoint-directory>
```

and the resulting model can be evaluated with

```
./scripts/test_models.sh <checkpoint-directory>
```

## Semantic Segmentation
We also use the same framework performing semantic segmentation. As per the surface normal estimation experiments, we include a docker configuration file to make getting dependencies easier. Before training a semantic segmentation model you will need to change the paths in the relevant YAML configuration file to point to where you have stored the pre-trained models and datasets. Once this is done the training script can be run with, e.g.,

```
python train.py --gpus 0,1 --cfg selfsupconfig/byol.yaml
```

where `selfsupconfig/byol.yaml` is the aforementioned configuration file. The resulting model can be evaluated with

```
python eval_multipro.py --gpus 0,1 --cfg selfsupconfig/byol.yaml
```

## Citation
If you find our work useful for your research, please consider citing our paper:
```bibtex
@misc{ericsson2020selfsupervised,
      title={How Well Do Self-Supervised Models Transfer?}, 
      author={Linus Ericsson and Henry Gouk and Timothy M. Hospedales},
      year={2020},
      eprint={2011.13377},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
If you have any questions, feel welcome to create an issue or contact Linus Ericsson (linus.ericsson@ed.ac.uk).
