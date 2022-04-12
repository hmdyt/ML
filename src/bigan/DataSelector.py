import torch
import torchvision
import argparse
import numpy as np

class DataSelector:
    def __init__(self, opt: argparse.Namespace):
        self._dataset_name = opt.dataset_name
        self._opt = opt
        self._select_dataset()
    
    def get_dataset(self):
        return self._train_data

    def get_loader(self):
        return self._train_loader
    
    def _select_dataset(self):
        if self._dataset_name == 'cifar10':
            i_class = 3
            self._load_cifar10(i_class)
        elif self._dataset_name == 'mnist':
            self._load_MNIST()
        else:
            raise ValueError('Dataset {} is not supported.'.format(self._dataset_name))

    def _load_MNIST(self):
        self._train_data = torchvision.datasets.MNIST(
            root = self._opt.dataset_dir,
            train = True,
            download = True,
            transform = torchvision.transforms.ToTensor(),
        )
        self._train_loader = torch.utils.data.DataLoader(
            self._train_data,
            batch_size = self._opt.batch_size,
            shuffle = True,
            num_workers = self._opt.num_worker,
        )
    
    def _load_cifar10(self, i_class: int = 3):
        self._train_data = torchvision.datasets.CIFAR10(
            root = self._opt.dataset_dir,
            train = True,
            download = True,
            transform = torchvision.transforms.ToTensor(),
        )
        sampler_mask = np.array([d[1] == i_class for d in self._train_data])
        sampler = torch.utils.data.sampler.WeightedRandomSampler(sampler_mask, len(sampler_mask))
        self._train_loader = torch.utils.data.DataLoader(
            self._train_data,
            batch_size = self._opt.batch_size,
            num_workers = self._opt.num_worker,
            sampler=sampler
        )
        print('Loaded {} dataset only with {} class.'.format(self._dataset_name, self._train_data.classes[i_class]))