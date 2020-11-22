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

## Many-shot evaluation
We provide the bare essentials of our evaluation in `essential_eval.py`.

To evaluate DeepCluster-v2 on CIFAR10 given our pre-computed best regularisation hyperparameter run:
```
python essential_eval.py --dataset cifar10 --model deepcluster-v2 --C 0.562
```
And for the Supervised baseline:
```
python essential_eval.py --dataset cifar10 --model supervised --C 0.056
```
