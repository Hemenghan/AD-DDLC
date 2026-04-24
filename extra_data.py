import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import hashlib
import urllib.request

NPZ_URL = "https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/1m.npz"
# NPZ_URL_100 = "https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/1m.npz"
NPZ_MD5 = "d65d45f6b6c6e0b5b3b8b0e0e0b0b0e0" 
from PIL import Image

def _download(extra_root="extra_data"):
    os.makedirs(extra_root, exist_ok=True)
    fpath = os.path.join(extra_root, "cifar10_ddpm.npz")
    if os.path.exists(fpath):
        return fpath
    print("[extra_data] downloading cifar10_ddpm.npz ...")
    urllib.request.urlretrieve(NPZ_URL, fpath)

    return fpath

class ExtraCIFAR10(Dataset):
    def __init__(self, transform=None, extra_root="/data/cifar10_1m.npz",smooth=0.1):

        npz = np.load(extra_root)
        self.data   = npz["image"]          
        self.labels = npz["label"]          
        self.transform = transform or T.ToTensor()
        self.smooth = smooth

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, lbl = self.data[idx], self.labels[idx]  
        img = Image.fromarray(img)                    
        img = self.transform(img)
        if self.smooth > 0:
            lbl = (1 - self.smooth) * lbl + self.smooth / 10
        return img, lbl

class InfiniteMixedLoader:

    def __init__(self, real_loader: DataLoader, extra_dataset: Dataset,
                 mix_ratio: float = 0.7,   
                 drop_last: bool = True):

        self.real_loader = real_loader
        self.mix_ratio = mix_ratio
        self.drop_last = drop_last

        total_bs = real_loader.batch_size
        extra_bs = int(total_bs * mix_ratio)
        real_bs = total_bs - extra_bs

        self.extra_loader = DataLoader(
            extra_dataset,
            batch_size=extra_bs,
            shuffle=True,
            num_workers=real_loader.num_workers,
            drop_last=drop_last,
            pin_memory=True
        )
        self.real_bs = real_bs
        self.extra_bs = extra_bs

        self.real_iter = iter(self.real_loader)
        self.extra_iter = iter(self.extra_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            real_img, real_lbl = next(self.real_iter)
        except StopIteration:
            self.real_iter = iter(self.real_loader)
            real_img, real_lbl = next(self.real_iter)

        try:
            extra_img, extra_lbl = next(self.extra_iter)
        except StopIteration:
            self.extra_iter = iter(self.extra_loader)
            extra_img, extra_lbl = next(self.extra_iter)

        img = torch.cat([real_img, extra_img], dim=0)
        lbl = torch.cat([real_lbl, extra_lbl], dim=0)

        perm = torch.randperm(img.size(0))
        return img[perm], lbl[perm]

    def __len__(self):
        return len(self.real_loader)

def mix_real_extra_loader(real_loader: DataLoader, mix_ratio: float = 0.7):

    extra_transform = real_loader.dataset.transform  
    extra_dataset = ExtraCIFAR10(transform=extra_transform,smooth=0.1)
    return InfiniteMixedLoader(real_loader, extra_dataset, mix_ratio=mix_ratio)