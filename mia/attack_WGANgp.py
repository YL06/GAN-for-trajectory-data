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

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='E:/different_sizes/first-stage-GAN/grid_data_forfirst/data_1000/', help='path to dataset')
parser.add_argument('--testroot', default='E:/different_sizes/MIA/data_1000/test_data/', help='path to testset')
parser.add_argument('--labelroot', default='./traj_all.txt', help='path to label')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='inputs batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the inputs image to network')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netD', default='E:/different_size/server/data_1000/netD_epoch_10001.pth', help="path to netD target discriminator")
parser.add_argument('--outf', default='E:/data/MIA/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--start_iter',type = int, default = 0)

opt = parser.parse_args()
print(opt)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

cudnn.benchmark = True

# check for cuda stuff
# print("cuda available:", torch.cuda.is_available())
# print(torch.__config__.show())
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

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


device = torch.device("cuda" if opt.cuda else "cpu")
ngpu = torch.cuda.device_count()
nz = int(opt.nz)
ndf = int(opt.ndf)
ngf = int(opt.ngf)
nc = 3
# Loss weight for gradient penalty
lambda_gp = 10



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.InstanceNorm = functools.partial(nn.InstanceNorm2d, affine=True)
        self.main = nn.Sequential(
        # state size. (ndf) x 32 x 32
        nn.Conv2d(nc, ndf * 1, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        nn.Conv2d(ndf * 1, ndf * 2, 4, 2, 1, bias=False),
        self.InstanceNorm(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),

        # state size. (ndf*4) x 8 x 8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        self.InstanceNorm(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 4 x 4
        nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
        )

    def forward(self, inputs):
        outputs = self.main(inputs)

        return outputs.view(-1, 1).squeeze(1)



def train(dataloader, netD, netG):
    wb_predictions = []
    # loop over real training_data
    for i, (real_imgs) in enumerate(dataloader):
        real_imgs = real_imgs.float()
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        real_validity = netD(real_imgs)
        # Fake images


def main():

    dataset = myImageFloder(root=opt.dataroot, label_root=opt.labelroot,
                            transform=transforms.Compose([transforms.ToTensor()]))
    testset = myImageFloder(root=opt.testroot, label_root=opt.labelroot,
                            transform=transforms.Compose([transforms.ToTensor()]))
    assert dataset
    print('Loading data')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize * ngpu,
                                             shuffle=True, num_workers=int(opt.workers), drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize * ngpu,
                                             shuffle=True, num_workers=int(opt.workers), drop_last=True)
    print('Preparing Model')
    netD = Discriminator()
    netD = torch.load(opt.netD)
    if opt.cuda:
        netD.cuda()
    if ngpu > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        netD = nn.DataParallel(netD, device_ids=[0,1,2,3])


    #netG.apply(weights_init)
    #if opt.netG != '':
    #    state_dict = torch.load(opt.netG).state_dict()
    #    netG.load_state_dict(state_dict)
    #netD.apply(weights_init)
    #if opt.netD != '':
    #    state_dict = torch.load(opt.netD).state_dict()
    #    netD.load_state_dict(state_dict)
    # setup optimizer
    #optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    #optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    print('Calculating')
    wb_predictions = []
    # loop over real training data
    for i, (real_imgs) in enumerate(dataloader):
        real_imgs = real_imgs.float()
        real_imgs = real_imgs.to(device)
        real_validity = netD(real_imgs)
        #("training data before loop", real_validity)
        real_validity = [x for x in real_validity.detach().cpu().numpy()]
        #print("training data", real_validity)
        real_validity = list(zip(real_validity, ['train' for _ in range(len(real_validity))]))
        wb_predictions.extend(real_validity)

    # loop over test data
    for i, (real_imgs) in enumerate(testloader):
        real_imgs = real_imgs.float()
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        real_validity = netD(real_imgs)
        real_validity = [x for x in real_validity.detach().cpu().numpy()]
        #print("test data", real_validity)
        real_validity = list(zip(real_validity, ['test' for _ in range(len(real_validity))]))
        wb_predictions.extend(real_validity)

    # get absolute number
    #wb_predictions = [abs(ele) for ele in wb_predictions]
    real_pred = [x for x in sorted(wb_predictions, reverse=False)[:len(dataset)]]
    wb_predictions = [x[1] for x in sorted(wb_predictions, reverse=True)[:len(dataset)]]
    wb_accuracy = wb_predictions.count('train') / float(len(dataset))


    print("baseline (random guess) accuracy: {:.3f}".format(len(dataset) / float(len(dataset) + len(testset))))
    print("white-box attack accuracy: {:.3f}".format(wb_accuracy))
if __name__ == '__main__':
    main()
