#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models

import PIL
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from temperature_scaling import cross_validate_temp_scaling, DummyDataset

from datasets.dtd import DTD
from datasets.pets import Pets
from datasets.cars import Cars
from datasets.food import Food
from datasets.sun397 import SUN397
from datasets.voc2007 import VOC2007
from datasets.flowers import Flowers
from datasets.aircraft import Aircraft
from datasets.caltech101 import Caltech101


def voc_ap(rec, prec):
    """
    average precision calculations for PASCAL VOC 2007 metric, 11-recall-point based AP
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :return: average precision
    """
    ap = 0.
    for t in np.linspace(0, 1, 11):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap += p / 11.
    return ap

def voc_eval_cls(y_true, y_pred):
    # get precision and recall
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    # compute average precision
    ap = voc_ap(rec, prec)
    return ap


# Testing classes and functions

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes, metric):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.metric = metric
        self.clf = LogReg(solver='lbfgs', multi_class='multinomial', warm_start=True)

        print('Logistic regression:')
        print(f'\t solver = L-BFGS')
        print(f"\t classes = {self.num_classes}")
        print(f"\t metric = {self.metric}")

    def set_params(self, d):
        self.clf.set_params(**d)

    @ignore_warnings(category=ConvergenceWarning)
    def fit_logistic_regression(self, X_train, y_train, X_test, y_test):
        if self.metric == 'accuracy':
            self.clf.fit(X_train, y_train)
            test_acc = 100. * self.clf.score(X_test, y_test)
            return test_acc

        elif self.metric == 'mean per-class accuracy':
            self.clf.fit(X_train, y_train)
            pred_test = self.clf.predict(X_test)

            #Get the confusion matrix
            cm = confusion_matrix(y_test, pred_test)
            cm = cm.diagonal() / cm.sum(axis=1) 
            test_score = 100. * cm.mean()

            return test_score

        elif self.metric == 'mAP':
            aps_test = []
            for cls in range(self.num_classes):
                self.clf.fit(X_train, y_train[:, cls])
                pred_test = self.clf.decision_function(X_test)
                ap_test = voc_eval_cls(y_test[:, cls], pred_test)
                aps_test.append(ap_test)

            mAP_test = 100. * np.mean(aps_test)

            return mAP_test

        else:
            raise Error(f'Metric {self.metric} not implemented')


class LinearTester():
    def __init__(self, model, train_loader, val_loader, trainval_loader, test_loader, batch_size, metric,
                 device, num_classes, feature_dim=2048, wd_range=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trainval_loader = trainval_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
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

    def get_features(self, train_loader, test_loader, model, test=True):
        X_train_feature, y_train = self._inference(train_loader, model, 'train')
        X_test_feature, y_test = self._inference(test_loader, model, 'test' if test else 'val')
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
            self.train_loader, self.val_loader, self.model, test=False
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
        test_score = self.classifier.fit_logistic_regression(X_trainval_feature, y_trainval, X_test_feature, y_test)

        orig_model = lambda x: torch.from_numpy(
            self.classifier.clf.decision_function(x.cpu().numpy())
        ).to(torch.float32)
        test_dataset = DummyDataset(X_test_feature, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        if self.metric != 'mAP':
            ece, scaled_ece = cross_validate_temp_scaling(orig_model, test_loader, self.batch_size)
        else:
            ece, scaled_ece = None, None

        return test_score, ece, scaled_ece, self.best_params['C']


class ResNetBackbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        self.model = models.resnet50(pretrained=False)
        del self.model.fc

        state_dict = torch.load(os.path.join('models', self.model_name + '.pth'))
        self.model.load_state_dict(state_dict)

        self.model.eval()
        print("Number of model parameters:", sum(p.numel() for p in self.model.parameters()))

    def forward(self, x):
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

    if dset in [Aircraft, DTD, Flowers, VOC2007]:
        # if we have a predefined validation set
        train_dataset = get_dataset(dset, data_dir, 'train', transform)
        valid_dataset = get_dataset(dset, data_dir, 'val', transform)
        trainval_dataset = ConcatDataset([train_dataset, valid_dataset])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        trainval_loader = DataLoader(
            trainval_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
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

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        trainval_loader = DataLoader(
            trainval_dataset, batch_size=batch_size, shuffle=shuffle,
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

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def prepare_data(dset, data_dir, batch_size, image_size, normalisation):
    if normalisation:
        normalise_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    else:
        normalise_dict = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
    train_loader, val_loader, trainval_loader = get_train_valid_loader(dset, data_dir, normalise_dict,
                                                batch_size, image_size, random_seed=0)
    test_loader = get_test_loader(dset, data_dir, normalise_dict, batch_size, image_size)

    return train_loader, val_loader, trainval_loader, test_loader


# name: {class, root, num_classes, metric}
LINEAR_DATASETS = {
    'aircraft': [Aircraft, '../data/Aircraft', 100, 'mean per-class accuracy'],
    'caltech101': [Caltech101, '../data/Caltech101', 102, 'mean per-class accuracy'],
    'cars': [Cars, '../data/Cars', 196, 'accuracy'],
    'cifar10': [datasets.CIFAR10, '../data/CIFAR10', 10, 'accuracy'],
    'cifar100': [datasets.CIFAR100, '../data/CIFAR100', 100, 'accuracy'],
    'dtd': [DTD, '../data/DTD', 47, 'accuracy'],
    'flowers': [Flowers, '../data/Flowers', 102, 'mean per-class accuracy'],
    'food': [Food, '../data/Food', 101, 'accuracy'],
    'pets': [Pets, '../data/Pets', 37, 'mean per-class accuracy'],
    'sun397': [SUN397, '../data/SUN397', 397, 'accuracy'],
    'voc2007': [VOC2007, '../data/VOC2007', 20, 'mAP'],
}

# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate pretrained self-supervised model via logistic regression.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', help='name of the dataset to evaluate on')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='the size of the mini-batches when inferring features')
    parser.add_argument('-i', '--image-size', type=int, default=224, help='the size of the input images')
    parser.add_argument('-w', '--wd-values', type=int, default=45, help='the number of weight decay values to validate')
    parser.add_argument('-c', '--C', type=float, default=None, help='sklearn C value (1 / weight_decay), if not tuning on validation set')
    parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    parser.add_argument('--device', type=str, default='cuda', help='CUDA or CPU training (cuda | cpu)')
    args = parser.parse_args()
    args.norm = not args.no_norm
    pprint(args)

    # load dataset
    dset, data_dir, num_classes, metric = LINEAR_DATASETS[args.dataset]
    train_loader, val_loader, trainval_loader, test_loader = prepare_data(
        dset, data_dir, args.batch_size, args.image_size, normalisation=args.norm)

    # load pretrained model
    model = ResNetBackbone(args.model)
    model = model.to(args.device)

    # evaluate model on dataset by fitting logistic regression
    tester = LinearTester(model, train_loader, val_loader, trainval_loader, test_loader, args.batch_size,
                          metric, args.device, num_classes, wd_range=torch.logspace(-6, 5, args.wd_values))
    if args.C is None:
        # tune hyperparameters
        tester.validate()
    else:
        # use the weight decay value supplied in arguments
        tester.best_params = {'C': args.C}
    # use best hyperparameters to finally evaluate the model
    test_acc, ece, scaled_ece, C = tester.evaluate()
    print(f'Final accuracy for {args.model} on {args.dataset}: {test_acc:.2f}% using hyperparameter C: {C:.3f}')
    print(f'ECE: {ece:.3f}, temperature scaled ECE: {scaled_ece:.3f}')
