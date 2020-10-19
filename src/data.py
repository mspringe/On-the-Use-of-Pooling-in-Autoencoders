"""
This module provides data laoder, that stores all data in RAM.
This approach is worthwhile for smaller datasets, as it bypasses an I/O bottleneck, 
but it is ill advised to use the data loader for data sets spanning more than a couple of Gb.
"""
import numpy as np
import torch
import cv2
from os.path import join, isdir
from os import listdir
from tqdm import tqdm
from colorama import Fore


class RAMDataSet(torch.utils.data.Dataset):
    """Dataset that loads all images into RAM, not recommendable for anything bigger than a couple of Gb"""

    def __init__(self, path, height, width, *args, flatten=False, inv=False, suffix='.png', norm_light=True,
                 preload=True, black_white=True):
        """
        :param path: path to images
        :param height: preffered height
        :param width: preffered width
        :param args: other arguments for the super-class :class:`torch.utils.data.Dataset`
        :param flatten: indicates whether to flatten the images
        :param inv: indicates whether to invert the images
        :param suffix: file ending of images, e.g. '.png'
        :param norm_light: indicates whether to normalize lighting
        :param preload: indicates imidiate preloading of data
        :param black_white: indicates whether to reduce batch and targets to black and white images
        """
        super().__init__(*args)
        self.print_c = Fore.YELLOW
        self.BW = black_white
        self.img_path = path
        self.norm_light = norm_light
        self.paths = self.__search_imgs(path, suffix=suffix)
        self.width = width
        self.height = height
        self.flatten = flatten
        self.inv = inv
        self.imgs = None
        if preload:
            self.load_all()

    def __getitem__(self, idx):
        """
        :param idx: image index
        :return: batch and targets tensors
        """
        if self.BW:
            if not isinstance(idx, int):
                return torch.from_numpy(self.imgs[idx]),\
                       torch.from_numpy(self.imgs[idx])
            return torch.from_numpy(self.imgs[idx].reshape(1, 1, *self.imgs.shape[2:])),\
                   torch.from_numpy(self.imgs[idx].reshape(1, 1, *self.imgs.shape[2:]))
        else:
            if not isinstance(idx, int):
                return torch.from_numpy(self.imgs[idx][:, 0, :, :][:, None, :, :]),\
                       torch.from_numpy(self.imgs[idx][:, 1:, :, :])
            return torch.from_numpy(self.imgs[idx][0, :, :].reshape(1, 1, *self.imgs.shape[2:])),\
                   torch.from_numpy(self.imgs[idx][1:, :, :].reshape(1, 2, *self.imgs.shape[2:]))

    def __len__(self):
        return self.imgs.shape[0]

    def __load_img(self, idx):
        return RAMDataSet.__load_img_static(self.paths[idx], self.height, self.width, self.norm_light,
                                            self.inv, self.flatten, self.BW)

    @staticmethod
    def __load_img_static(img_path, height, width, norm_light, inv, flatten, BW=True):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (height, width))
        if BW:
            img = img.mean(axis=2).astype(np.uint8)
            img = img.reshape(1, height, width).astype(np.float32) / 255.
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.
        if BW and norm_light:
                img -= img.min()
                img /= img.max()
        if BW and inv:
            img = 1.0-img
        if flatten:
            return img.flatten()
        return img

    def __search_imgs(self, path, suffix='.png'):
        paths = []
        for p in listdir(path):
            if isdir(join(path, p)):
                paths += self.__search_imgs(join(path, p), suffix)
            elif p.endswith(suffix):
                paths.append(join(path, p))
        return paths

    def load_all(self):
        """Loading all images into RAM."""
        n_fmaps = 1 if self.BW else 3
        if not self.flatten:
            self.imgs = np.empty((len(self.paths), n_fmaps, self.height, self.width), dtype=np.float32)
        else:
            self.imgs = np.empty((len(self.paths), n_fmaps * self.height * self.width), dtype=np.float32)
        for i in tqdm(range(len(self)), bar_format=f'{self.print_c}preloading data set{Fore.RESET}    '+
                                                   "{l_bar}%s{bar}%s{r_bar}" % (self.print_c, Fore.RESET)):
            self.imgs[i] = self.__load_img(i)


class RAMDataSetIter:
    """Iterator for RAMDataSet, does not need reinitialization after an epoch has finished (can be reset)."""

    def __init__(self, dset, shuffle=True, s_batch=16):
        """
        :param dset: data set to iterate on
        :param shuffle: indicates whether to shuffle the data
        :param s_batch: size of batch
        """
        self.dset = dset
        self.s_batch = s_batch
        self.idcs = np.arange(len(dset))
        self.shuffle = shuffle
        if self.shuffle:
            self.__shuffle()
        self.__idx = 0

    def __next__(self):
        if self.__idx >= len(self.dset):
            raise StopIteration
        self.__idx += self.s_batch
        idcs = self.idcs[self.__idx-self.s_batch:min(len(self.dset), self.__idx)]
        return self.dset[idcs]

    def __shuffle(self):
        self.idcs = np.random.permutation(self.idcs)

    def next(self):
        """
        :return: next batch and targets from dataset
        """
        return self.__next__()

    def reset(self):
        """Resetting the iterator and shuffling again, if shuffling was specified."""
        self.__idx = 0
        if self.shuffle:
            self.__shuffle()
