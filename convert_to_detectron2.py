#!/usr/bin/env python
# adapted from the conversion code in `detectron2/tools/convert-torchvision-to-d2.py`

import sys, os
import pickle as pkl

import torch

def convert_to_detectron(model_name, pytorch_dir, detectron_dir):
    """
    This function will convert a model from our
    common PyTorch format into the detectron2 format.
    """
    print(f'Converting {model_name}')
    input_path = f'{pytorch_dir}/{model_name}.pth'
    output_path = f'{detectron_dir}/{model_name}.pkl'

    try :
        obj = torch.load(input_path, map_location="cpu")

        newmodel = {}
        for k in list(obj.keys()):
            old_k = k
            if "layer" not in k:
                k = "stem." + k
            for t in [1, 2, 3, 4]:
                k = k.replace("layer{}".format(t), "res{}".format(t + 1))
            for t in [1, 2, 3]:
                k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
            k = k.replace("downsample.0", "shortcut")
            k = k.replace("downsample.1", "shortcut.norm")
            #print(old_k, "->", k)
            newmodel[k] = obj.pop(old_k).detach().numpy()

        res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

        with open(output_path, "wb") as f:
            pkl.dump(res, f)
        if obj:
            print("Unconverted keys:", obj.keys())
    except FileNotFoundError as e:
        print(f'Could not find model at {input_path}. Skipping {model_name}.')

MODELS = [
    'supervised',
    'simclr-v1',
    'simclr-v2',
    'moco-v1',
    'moco-v2',
    'byol',
    'swav',
    'deepcluster-v2',
    'sela-v2',
    'infomin',
    'insdis',
    'pirl',
    'pcl-v1',
    'pcl-v2'
]

pytorch_dir = 'models'
detectron_dir = 'models/detectron2'

os.makedirs(pytorch_dir, exist_ok=True)
os.makedirs(detectron_dir, exist_ok=True)

for model_name in MODELS:
    convert_to_detectron(model_name, pytorch_dir, detectron_dir)
