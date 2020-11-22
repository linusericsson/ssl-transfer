#!/usr/bin/env python
# coding: utf-8

import sys, os
import argparse
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models

import PIL
import numpy as np
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes, metric):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.metric = metric
        self.clf = LogReg(solver='lbfgs', multi_class='multinomial', warm_start=True)

        print('Logistic regression:', flush=True)
        print(f'\t solver = L-BFGS', flush=True)
        print(f"\t classes = {self.num_classes}", flush=True)
        print(f"\t metric = {self.metric}", flush=True)

    def set_params(self, d):
        self.clf.set_params(**d)

    @ignore_warnings(category=ConvergenceWarning)
    def fit_logistic_regression(self, X_train, y_train, X_test, y_test):
        self.clf.fit(X_train, y_train)
        train_acc = 100. * self.clf.score(X_train, y_train)
        test_acc = 100. * self.clf.score(X_test, y_test)
        return test_acc


class LinearTester():
    def __init__(self, model, train_loader, val_loader, trainval_loader, test_loader, metric,
                 device, num_classes, feature_dim=2048, wd_range=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trainval_loader = trainval_loader
        self.test_loader = test_loader
        self.metric = metric
        self.device = device
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.best_params = {}

        if wd_range is None:
            self.wd_range = torch.logspace(-6, 5, 45)
        else:
            self.wd_range = wd_range

        self.classifier = LogisticRegression(self.feature_dim, self.num_classes, self.metric).to(self.device)

    def get_features(self, train_loader, test_loader, model):
        X_train_feature, y_train = self._inference(train_loader, model, 'train')
        X_test_feature, y_test = self._inference(test_loader, model, 'test')
        return X_train_feature, y_train, X_test_feature, y_test

    def _inference(self, loader, model, split):
        model.eval()
        feature_vector = []
        labels_vector = []
        for data in tqdm(loader, desc=f'Computing features for {split} set'):
            batch_x, batch_y = data
            batch_x = batch_x.to(self.device)
            labels_vector.extend(np.array(batch_y))

            features = model(batch_x)
            feature_vector.extend(features.cpu().detach().numpy())

        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector, dtype=int)

        return feature_vector, labels_vector

    def validate(self):
        X_train_feature, y_train, X_val_feature, y_val = self.get_features(
            self.train_loader, self.val_loader, self.model
        )
        best_score = 0
        for wd in tqdm(self.wd_range, desc='Selecting best hyperparameters'):
            C = 1. / wd.item()
            self.classifier.set_params({'C': C})
            test_score = self.classifier.fit_logistic_regression(X_train_feature, y_train, X_val_feature, y_val)

            if test_score > best_score:
                best_score = test_score
                self.best_params['C'] = C

    def evaluate(self):
        print(f"Best hyperparameters {self.best_params}")
        X_trainval_feature, y_trainval, X_test_feature, y_test = self.get_features(
            self.trainval_loader, self.test_loader, self.model
        )
        self.classifier.set_params({'C': self.best_params['C']})
        return self.classifier.fit_logistic_regression(X_trainval_feature, y_trainval, X_test_feature, y_test)


class LoadedResNet(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        if model_name == 'supervised':
            self.model = models.resnet50(pretrained=True)
            del self.model.fc
        else:
            self.model = models.resnet50(pretrained=False)
            del self.model.fc

            state_dict = torch.load(os.path.join('models', self.model_name + '.pth'))
            self.model.load_state_dict(state_dict)

        self.model.train()
        print("num parameters:", sum(p.numel() for p in self.model.parameters()))

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x

def load_pretrained_model(model_name):
    model = LoadedResNet(model_name)
    return model


# Data classes and functions

def get_dataset(dset, root, split, transform):
    try:
        return dset(root, train=(split == 'train'), transform=transform, download=True)
    except:
        return dset(root, split=split, transform=transform, download=True)


def get_train_valid_loader(dset,
                           data_dir,
                           normalise_dict,
                           batch_size,
                           image_size,
                           random_seed,
                           valid_size=0.2,
                           shuffle=True,
                           num_workers=1,
                           pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - dset: dataset class to load.
    - data_dir: path directory to the dataset.
    - normalise_dict: dictionary containing the normalisation parameters.
    - batch_size: how many samples per batch to load.
    - image_size: size of images after transforms.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    - trainval_loader: iterator for the training and validation sets combined.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(**normalise_dict)
    print("Train normaliser:", normalize)

    # define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # otherwise we select a random subset of the train set to form the validation set
    train_dataset = get_dataset(dset, data_dir, 'train', transform)
    valid_dataset = get_dataset(dset, data_dir, 'train', transform)
    trainval_dataset = get_dataset(dset, data_dir, 'train', transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    trainval_loader = torch.utils.data.DataLoader(
        trainval_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, trainval_loader


def get_test_loader(dset,
                    data_dir,
                    normalise_dict,
                    batch_size,
                    image_size,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - dset: dataset class to load.
    - data_dir: path directory to the dataset.
    - normalise_dict: dictionary containing the normalisation parameters.
    - batch_size: how many samples per batch to load.
    - image_size: size of images after transforms.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """

    normalize = transforms.Normalize(**normalise_dict)
    print("Test normaliser:", normalize)

    # define transform
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = get_dataset(dset, data_dir, 'test', transform)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def prepare_data(dset, data_dir, batch_size, image_size, normalisation):
    if normalisation == 'imagenet':
        normalise_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    elif normalisation == 'none':
        normalise_dict = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
    else:
        raise ValueError
    train_loader, val_loader, trainval_loader = get_train_valid_loader(dset, data_dir, normalise_dict,
                                                batch_size, image_size, random_seed=0)
    test_loader = get_test_loader(dset, data_dir, normalise_dict, batch_size, image_size)

    return train_loader, val_loader, trainval_loader, test_loader


# name: {url, format}
MODELS = {
    'insdis': ['https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACcsSIt1_Njv7GsmsuzZ6Sta/InsDis.pth', 'pth'],
    'moco-v1': ['https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar', 'pth'],
    'pcl-v1': ['https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v1_epoch200.pth.tar', 'pth'],
    'pirl': ['https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth', 'pth'],
    'pcl-v2': ['https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v2_epoch200.pth.tar', 'pth'],
    'simclr-v1': ['', ''],
    'moco-v2': ['https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar', 'pth'],
    'simclr-v2': ['', ''],
    'sela-v2': ['https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar', 'pth'],
    'infomin': ['https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAAzMTynP3Qc8mIE4XWkgILUa/InfoMin_800.pth', 'pth'],
    'byol': ['https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl', 'pkl'],
    'deepcluster-v2': ['https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar', 'pth'],
    'swav': ['https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar', 'pth'],
    'supervised-simclr': ['', ''],
    'supervised': ['', ''],
}

# name: {class, root, num_classes, metric}
MANY_SHOT_DATASETS = {
    'cifar10': [datasets.CIFAR10, '../data/CIFAR10', 10, 'accuracy'],
    'cifar100': [datasets.CIFAR100, '../data/CIFAR100', 100, 'accuracy'],
}

# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate pretrained self-supervised model with logistic regression.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', help='name of the dataset to evaluate on')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='the size of the mini-batches when inferring features')
    parser.add_argument('-i', '--image-size', type=int, default=224, help='the size of the input images')
    parser.add_argument('-w', '--wd-values', type=int, default=45, help='the number of weight decay values to validate')
    parser.add_argument('-c', '--C', type=float, default=None, help='sklearn C value (1 / weight_decay), if not tuning on validation set')
    parser.add_argument('-n', '--norm', type=str, default='imagenet', help='data normalisation (none | imagenet)')
    parser.add_argument('--device', type=str, default='cuda', help='CUDA or CPU training (cuda | cpu)')
    args = parser.parse_args()
    pprint(args)

    # load dataset - this file only supports CIFAR10/CIFAR100
    # see eval.py for full range of many-shot and few-shot datasets.
    dset, data_dir, num_classes, metric = MANY_SHOT_DATASETS[args.dataset]
    train_loader, val_loader, trainval_loader, test_loader = prepare_data(
        dset, data_dir, args.batch_size, args.image_size, normalisation=args.norm)

    # load pretrained model - 
    model = load_pretrained_model(args.model)
    model = model.to(args.device)

    # evaluate model on dataset by fitting logistic regression
    tester = LinearTester(model, train_loader, val_loader, trainval_loader, test_loader,
                          metric, args.device, num_classes, wd_range=torch.logspace(-6, 5, args.wd_values))
    if args.C is None:
        # tune hyperparameters
        tester.validate()
    else:
        # use the weight decay value supplied in arguments
        tester.best_params = {'C': args.C}
    # use best hyperparameters to finally evaluate the model
    test_acc = tester.evaluate()
    print(f'Final accuracy for {args.model} on {args.dataset}: {test_acc:.2f}%')
