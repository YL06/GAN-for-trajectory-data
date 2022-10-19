import argparse
import os
import random
import numpy as np
import functools
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from tensorboardX import SummaryWriter

dataroot = 'D:/YJ/single_trj_data/grid32/'
labelroot = './traj_all.txt'
batchSize = 64
workers = 4
ngpu = 4


def default_loader(path):
    return np.load(path)


class myImageFloder(data.Dataset):
    def __init__(self,root,label_root,target_transform = None,transform=None,
                loader = default_loader):
        fh = open(label_root)
        imgs = []
        for line in fh.readlines():
            cls = line.split()
            fn = cls.pop(0)
            if os.path.isfile(os.path.join(root,fn)):
                imgs.append(fn)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self,index):
        fn = self.imgs[index]
        img = self.loader(os.path.join(self.root,fn))
        if self.transform:
            img = self.gen_sample(img)
        return img

    def __len__(self):
        return len(self.imgs)

    def gen_sample(self, img):
        img = img.astype(np.float)
        img = img.transpose((1,2,0))
        img /= 795.0
        img -= [0.5, 0.5, 0.5]#[0.172185785,0.033457624, 0.003309033]
        img /= [0.5, 0.5, 0.5]
        img = img.transpose((2,0,1))
        img = torch.tensor(img)
        return img


device = torch.device("cuda")




def main():
    dataset = myImageFloder(root = dataroot,label_root = labelroot,
                            transform = transforms.Compose([transforms.ToTensor()]))
    assert dataset
    print('Loading data')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                             shuffle=True, num_workers=int(workers), drop_last = True)
    print("Loaded data")
    for i, (real_imgs) in enumerate(dataloader):
        real_imgs = real_imgs.float()

        # configure input
        real_imgs = real_imgs.to(device)

        # Get real imgs batch size
        batch_size = real_imgs.size(0)

        vutils.save_image((real_imgs/2+0.5)*795,
                            f'./real/real_samples_{i}.png',
                        normalize=True)
        vutils.save_image((real_imgs/2+0.5)*795,f'./real/real_samples_false_{i}.png',normalize=False)


if __name__ == '__main__':
    main()
