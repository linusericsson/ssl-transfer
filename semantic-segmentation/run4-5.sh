#!/bin/bash

# byol.pth  deepcluster-v2.pth  infomin.pth  insdis.pth  moco-v1.pth  moco-v2.pth  pcl-v1.pth  pcl-v2.pth  pirl.pth  sela-v2.pth	simclr-v1.pth  simclr-v2.pth  supervised.pth  supervised-simclr.pth  swav.pth

export CUDA_VISIBLE_DEVICES=4,5
#python train.py --gpus 0,1 --cfg selfsupconfig/pirl.yaml
#python train.py --gpus 0,1 --cfg selfsupconfig/sela-v2.yaml
python train.py --gpus 0,1 --cfg selfsupconfig/insdis.yaml

