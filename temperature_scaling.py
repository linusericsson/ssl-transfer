import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

from netcal.scaling import TemperatureScaling
from netcal.metrics import ECE


class DummyDataset(Dataset):
    """Dummy dataset."""

    def __init__(self, data, labels, transform=None, target_transform=None):
        """
        Args:
            data (tensor): datapoints for the dataset.
            labels (string): labels for the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)

        return sample, label


def rollout_loader(model, loader):
    """
    get model probabilities and labels from dataloader
    """
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for input, label in loader:
            input = input.cuda()
            logits = model(input)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs.numpy())
            labels_list.append(label.numpy())
        probs = np.concatenate(probs_list)
        labels = np.concatenate(labels_list)
    return probs, labels


def calibrate(model, valid_loader, test_loader, n_bins=15):
    """
    Calibrate the model via temperature scaling
    """
    confidence, labels = rollout_loader(model, valid_loader)
    test_confidence, test_labels = rollout_loader(model, test_loader)

    temperature = TemperatureScaling()
    temperature.fit(confidence, labels)
    calibrated = temperature.transform(test_confidence)

    ece = ECE(n_bins)
    calibrated_score = ece.measure(calibrated, test_labels)

    return calibrated_score


def cross_validate_temp_scaling(model, data_loader, batch_size, k=5, seed=0, num_workers=0, n_bins=15, pin_memory=False):
    """
    Perform temperature scaling on the model with k-fold cross validation
    """
    print("Computing model calibration", flush=True)
    test_dataset = data_loader.dataset
    num_test = len(test_dataset)
    indices = list(range(num_test))
    np.random.seed(seed)
    np.random.shuffle(indices)
    idxs = torch.tensor(indices).split(int(len(indices) / k))[:k]

    # get the uncalibrated ECE
    confidence, labels = rollout_loader(model, data_loader)
    ece = ECE(n_bins)
    unscaled_ece = ece.measure(confidence, labels)
    print(f'ECE: {unscaled_ece:.3f}')

    # compute the calibrated ECE
    scaled_eces = []
    # for each of the k folds
    for i in range(k):
        valid_idx = idxs[i]
        before = torch.cat(idxs[:i]) if i is not 0 else torch.tensor([], dtype=torch.long)
        after = torch.cat(idxs[i + 1:]) if i + 1 is not k else torch.tensor([], dtype=torch.long)
        test_idx = torch.cat([before, after])

        # create data loaders
        test_sampler = SubsetRandomSampler(test_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, sampler=test_sampler,
            num_workers=num_workers, pin_memory=pin_memory
        )
        valid_loader = DataLoader(
            test_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory
        )

        scaled_ece = calibrate(model, valid_loader, test_loader, n_bins)
        print(f'Cross validation fold {i}, temperature scaled ECE: {scaled_ece:.3f}')
        scaled_eces.append(scaled_ece)
    mean_scaled_ece = np.mean(scaled_eces)

    return unscaled_ece, mean_scaled_ece
