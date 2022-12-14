#coding:utf-8
import time
import torch.utils.data as Data
import torch.nn as nn
import random
import argparse
from torch.nn.utils.rnn import pack_padded_sequence
from model import Encoder, Decoder, Discriminator
from dataset import *
from utils import *
from vis import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
torch.backends.cudnn.benchmark = True
CUDA_VISIBLE_DEVICES = 0

ananas_size = 32

torch.manual_seed(0)
torch.cuda.manual_seed(0)
parser = argparse.ArgumentParser()
#E:/different_sizes/second-stage-GAN/input/data_1000
parser.add_argument('--data_path',type = str, default = 'home/lee/data/data_5000/')
parser.add_argument('--start_epoch',type = int, default = 1)
# from 120 to 60
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
# changed batch size from 32 because of cuda out of memory ananas
parser.add_argument('--batch_size', type=int, default=ananas_size, help='inputs batch size')
parser.add_argument('--workers', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate, default=1e-4')
parser.add_argument('--decoder_lr', type=float, default=4e-4, help='learning rate, default=1e-4')
parser.add_argument('--D_lr', type=float, default=4e-4, help='learning rate, default=1e-4')# clip gradients at an absolute value of
parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of, default=5')
parser.add_argument('--save_path', type=str, default='./checkpoint_rnn/', help='save path')  #vis_dir
parser.add_argument('--vis_dir', type=str, default='./vis/', help='vis path')  #
parser.add_argument('--loss', type=str, default='d', help='type of loss')  #
parser.add_argument('--gan', type=bool, default=False, help='type of decoder')  #

#leej 0420
float_formatter = "{:12.6f}".format
np.set_printoptions(threshold=1024, linewidth=160, formatter={'float_kind':float_formatter})
#'C:/Users/info/YJ/Second_stage_gan/train_res5000.txt'
out_fn = 'home/lee/train_res5000.txt'
res_file = open(out_fn, 'w')

opt = parser.parse_args()
print(opt)
data_path = opt.data_path
decoder_dim = 1024
dropout = 0.5

start_epoch = opt.start_epoch
epochs = opt.epochs  # number of epochs to train for (if early stopping is not triggered)
epochs_since_G_improvement = 0  # keeps track of number of epochs since there's been an improvement
epochs_since_D_improvement = 0
batch_size = opt.batch_size
workers = opt.workers  # for data-loading; right now, only 1 works with h5py
encoder_lr = opt.encoder_lr # learning rate for encoder if fine-tuning
decoder_lr = opt.decoder_lr  # learning rate for decoder
D_lr = opt.D_lr
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
g_best_loss = 1.  # best loss score right now
d_best_loss = 1.
print_freq = 1  # print training/validation stats every __ batches
fine_tune_encoder = False # fine-tune encoder?
checkpoint = None #'P:/BA/TSG/TSG/Second_stage_gan/checkpoint_rnn/checkpoint_epoch_17.pth' #None  # path to checkpoint, None if none
save_path = opt.save_path # checkpoint save path
vis_dir = opt.vis_dir # store visualized result
n_critic = 5

max_len = 12 # the longest sequence
EPSILON = 1e-40
# calculate
lambd = 1.
convsize = 7
std = 5


class DLoss(nn.Module):
    ''' C-RNN-GAN discriminator loss
    '''
    def __init__(self):
        super(DLoss, self).__init__()

    def forward(self, logits_real, logits_gen):
        logits_real = torch.clamp(logits_real, EPSILON, 1.0)
        d_loss_real = -torch.log(logits_real)
        logits_gen = torch.clamp((1 - logits_gen), EPSILON, 1.0)
        d_loss_gen = -torch.log(logits_gen)
        batch_loss = d_loss_real + d_loss_gen

        return torch.mean(batch_loss)
class GLoss(nn.Module):
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, logits_gen):
        logits_gen = torch.clamp(logits_gen, EPSILON, 1.0)
        batch_loss = -torch.log(logits_gen)
        return torch.mean(batch_loss)


def train(train_loader, encoder, decoder, D, criterion, encoder_optimizer, decoder_optimizer, D_optimizer, epoch, lambd, convsize, std, writer):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    encoder.train()
    decoder.train()
    D.train()


    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    G_losses = AverageMeter()  # generator's loss
    D_losses = AverageMeter()  # discriminator's loss
    #wayslosses = AverageMeter()

    start = time.time()

    for i,data in enumerate(train_loader):

        img_name = data['name']
        imgs = data['image'].to(device) # (b,c,w,h)
        seq = data['seq'].to(device) # (b,max_len,2)
        seq_inv = data['seq_inv'].to(device)
        enter = data['enter'].to(device) # (b,2)
        esc = data['esc'].to(device) # (b,4) one-hot indicate four direction
        length = data['len'] # (b) it seem to be a 1D CPU int64 tensor when use pack_padded_sequence below


        #if i in [1, 23, 24]:
        #    print("[{}] img_name:{} seq:{} seq_inv:{} enter:{} exit:{} len:{}".format(i, img_name, seq, seq_inv, enter, esc, length), file=res_file)

        # why skip?
        skip = [95,114,115,118,121, 123,212,214,221, 247,258,259, 262,265]


        if i in skip:
            continue
        data_time.update(time.time() - start)

        # Forward prop.
        imgs = encoder(imgs) # encoder_out

        #added 1 for teach_rate
        pred, pred_inv, pred_assemble, sort_ind = decoder(imgs, enter, esc, seq[:,:-1,:], seq_inv[:,:-1,:], length-1, 1)
        #print("pred", type(pred), pred.size())
        #print("pred_inv", type(pred_inv), pred_inv.size())
        #print("sort_ind", type(sort_ind), sort_ind.size())
        #print("pred", pred)
        #print("sort_ind", sort_ind)
        if i % 50 == 0:
            print("pred_assemble({}) shape:{} val:{}".format(i, pred_assemble.shape, pred_assemble), file=res_file)
        # pred (b,max_len,2)
        #if i in [4, 23, 24]:
        #    print("[{}] pred:: shape={} val={} pred_inv:: shape={} val={} pred_assemble:: shape={} val={}".format(i, pred.shape, pred, pred_inv.shape, pred_inv, pred_assemble.shape, pred_assemble), file=res_file)

        targets = seq[sort_ind,1:,:] # to the sorted version
        targets_inv = seq_inv[sort_ind,1:,:]
        #print("[{}] targets:{} targets_inv:{}".format(i, targets, targets_inv), file=res_file)

        # Remove timesteps that we didn't decode at, or are pads
        #pred = pack_padded_sequence(pred, length.squeeze(1), batch_first=True)
        #targets = pack_padded_sequence(targets, length.squeeze(1), batch_first=True)

        # used to calculate the loss of coordinates away from ways
        #reference = imgs.detach().permute(0,3,1,2) # (b, 1, encoded_image_size, encoded_image_size)
        #waysloss = cal_waysloss(reference, pred, pred_inv, convsize, std, device)
        # Calculate loss

        #+ lambd * waysloss
        #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        #-----------------
        # train generator
        #-----------------
        time_step = epoch*317+i
        if i%4==0:#i % n_critic:
            logits_gen, pred_grid= D(pred_assemble, imgs.detach())
            #errG = criterion['g'](logits_gen)#+nn.MSELoss(reduction='mean')(pred_assemble, seq)*0.5
            # label down to 32 from 128 because of batch size ananas
            label = torch.full((ananas_size*4*4,), 1, device=device)
            errG = nn.BCELoss()(logits_gen.view(-1), label)#+nn.MSELoss(reduction='mean')(pred_assemble, seq)*0.5

            decoder_optimizer.zero_grad()
            #if encoder_optimizer is not None:
            #    encoder_optimizer.zero_grad()

            errG.backward()

            # Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
            #    if encoder_optimizer is not None:
            #        clip_gradient(encoder_optimizer, grad_clip)

            # Update weights
            decoder_optimizer.step()
            #if encoder_optimizer is not None:
            #    encoder_optimizer.step()
            G_losses.update(errG.item(), length.sum().item())
            writer.add_images('pred_grid',np.reshape(pred_grid, (-1, 32, 32,1)),time_step,dataformats='NHWC')

#-------
        # else:
        logits_real, real_grid= D(seq.detach(), imgs.detach())

        logits_gen_d, pred_grid = D(pred_assemble.detach(), imgs.detach())
        # logits_gen_d [8,16]
        # view 128
        # errD = criterion['d'](logits_real, logits_gen_d)
        # 128 to 32 due to batch size ananas
        label_real = torch.full((ananas_size*4*4,), 1, device=device)
        label_fake = torch.full((ananas_size*4*4,), 0, device=device)
        loss_real = nn.BCELoss()(logits_real.view(-1),label_real)
        loss_fake = nn.BCELoss()(logits_gen_d.view(-1),label_fake)
        # added /2
        errD = (loss_real+loss_fake)/2
        print("dic real", loss_real)
        print("loss fake", loss_fake)

        D_optimizer.zero_grad()
        errD.backward()
        D_optimizer.step()
        # Keep track of metrics
        D_losses.update(errD.item(), length.sum().item())
        writer.add_images('real_grid',np.reshape(real_grid, (-1, 32, 32,1)),time_step, dataformats='NHWC')
        writer.add_images('pred_grid',np.reshape(pred_grid, (-1, 32, 32,1)),time_step,dataformats='NHWC')

#-------
        #wayslosses.update(waysloss.item(), length.sum().item())
        batch_time.update(time.time() - start)

        start = time.time()

        writer.add_scalar('loss_D', D_losses.val,time_step )
        writer.add_scalar('loss_G', G_losses.val, time_step)

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}] [{1}/{2}]\n'
                  'Batch Time {batch_time.val:.3f}s (Average:{batch_time.avg:.3f}s)\n'
                  'Data Load Time {data_time.val:.3f}s (Average:{data_time.avg:.3f}s)\n'
                  'G_Loss {g_loss.val:.4f} (Average:{g_loss.avg:.4f})\n'
                  'D_Loss {d_loss.val:.4f} (Average:{d_loss.avg:.4f})\n'
                  .format(epoch, i, len(train_loader),batch_time=batch_time,
                          data_time=data_time, g_loss = G_losses, d_loss = D_losses))

#             'waysloss {waysloss.val:.4f} (Average:{waysloss.avg:.4f})\n'


def validate(val_loader, encoder, decoder, D, criterion, lambd, convsize, std, device):
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    G_losses = AverageMeter()  # generator's loss
    D_losses = AverageMeter()  # discriminator's loss
    #losses = AverageMeter()
    #wayslosses = AverageMeter()

    start = time.time()

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        for i, data in enumerate(val_loader):
            print("for loop")
            print("i before skip", i)
            # Move to device, if available
            #why skip these deleted 0
            #skip = [1,5]

            #if i in skip:
            #    continue
            #print("i after skip", i)
            imgs = data['image'].to(device)  # (b,c,w,h)
            seq = data['seq'].to(device)  # (b,max_len,2)
            seq_inv = data['seq_inv'].to(device)
            enter = data['enter'].to(device)  # (b,2)
            esc = data['esc'].to(device)  # (b,4)
            length = data['len']  # (b)  it seem to be a 1D CPU int64 tensor
            # Forward prop.
            if encoder is not None:
                imgs_encode = encoder(imgs)
            # added teach_rate = 1
            pred, pred_inv,predictions_assemble, sort_ind = decoder(imgs_encode, enter, esc, seq[:,:-1,:], seq_inv[:,:-1,:], length - 1, 1)
            print("sort_ind", type(sort_ind), sort_ind.size(), sort_ind)

            targets = seq[sort_ind,1:,:]
            targets_inv = seq_inv[sort_ind,1:,:]

            #pred_cal = pred.clone()
            #pred_cal = pack_padded_sequence(pred_cal, length.squeeze(1), batch_first=True)
            #targets = pack_padded_sequence(targets, length.squeeze(1), batch_first=True)

            #reference = imgs_encode.detach().permute(0,3,1,2) # (b, 1,encoded_image_size, encoded_image_size)
            #waysloss = cal_waysloss(reference, pred, pred_inv, convsize, std, device)
            # Calculate loss
            # fake_loss = -*(1-D(predictions_assemble, imgs_encode)[0])*torch.log(1-D(predictions_assemble, imgs_encode)[0])
            # real_loss = -D(seq, imgs_encode)[0]*torch.log(D(seq, imgs_encode)[0])
            fake_loss = -(1-D(predictions_assemble, imgs_encode)[0])*torch.log(1-D(predictions_assemble, imgs_encode)[0])
            real_loss = -D(seq, imgs_encode)[0]*torch.log(D(seq, imgs_encode)[0])
            errD = torch.mean(real_loss) + torch.mean(fake_loss)

        #+ lambd * waysloss
        #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.

            #-----------------
            # train generator
                #-----------------
            if i % n_critic:
                errG = torch.mean(fake_loss)

                G_losses.update(errG.item(), length.sum().item())

            # Keep track of metrics
            D_losses.update(errD.item(), length.sum().item())
                #wayslosses.update(waysloss.item(), length.sum().item())
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
        #loss = criterion(pred, targets) + criterion(pred_inv, targets_inv)
        #+ lambd * waysloss

        # Keep track of metrics
        #losses.update(loss.item(),length.sum().item())
        #wayslosses.update(waysloss.item(),length.sum().item())
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\n'
                      'Batch Time {batch_time.val:.3f}s (Average:{batch_time.avg:.3f}s)\n'
                      'G_Loss {g_loss.val:.4f} (Average:{g_loss.avg:.4f})\n'
                      'D_Loss {d_loss.val:.4f} (Average:{d_loss.avg:.4f})\n'
                      .format(i, len(val_loader), batch_time=batch_time,g_loss=G_losses,d_loss = D_losses))
# 'waysloss {waysloss.val:.4f} (Average:{waysloss.avg:.4f})\n'

    # ab imgs nur noch f??r visualisierung ben??tigt
    return G_losses.avg, D_losses.avg, imgs[sort_ind,:,:,:], pred, predictions_assemble, enter[sort_ind,:], esc[sort_ind,:], length[sort_ind,:]

def main():
    global epochs_since_G_improvement, epochs_since_D_improvement, checkpoint, start_epoch, fine_tune_encoder, best_loss, save_path, vis_dir, decoder_dim, lambd, convsize, std

    if checkpoint is None:
        decoder = Decoder(decoder_dim, gan = opt.gan)
        #decoder = Decoder(decoder_dim)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr) if fine_tune_encoder else None
        D = Discriminator()
        D_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, D.parameters()),
                                       lr=encoder_lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        #added g and d
        epochs_since_G_improvement = checkpoint['epochs_since_G_improvement']
        epochs_since_G_improvement = checkpoint['epochs_since_D_improvement']
        G_best_loss = checkpoint['G_Loss']
        D_best_loss = checkpoint['D_Loss']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        D = checkpoint['Discriminator']
        D_optimizer = checkpoint['discriminator_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr)

    decoder = decoder.to(device)
    encoder = encoder.to(device)
    D = D.to(device)


    #criterion = nn.MSELoss().to(device)
    criterion = {'d':DLoss().to(device), 'g':GLoss().to(device)}
    #criterion = traj_loss().to(device)

    dataset = GuiderDataset(data_path,0.2,max_len=max_len)
    train_loader = Data.DataLoader(dataset.train_set(), batch_size=batch_size, shuffle=False,drop_last = True)
    val_loader = Data.DataLoader(dataset.test_set(), batch_size=batch_size, shuffle=False,drop_last = True)

    for epoch in range(start_epoch, start_epoch + epochs):
        writer = SummaryWriter(log_dir='log2/log')
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_G_improvement == 20:
            break
        if epochs_since_G_improvement > 0 and epochs_since_G_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              D = D,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              D_optimizer = D_optimizer,
              epoch=epoch, lambd=lambd, convsize=convsize, std=std, writer = writer)

        # One epoch's validation, return the average loss of each batch in this epoch
        G_loss, D_loss, imgs, pred, pred_vis, enter, esc, length = validate(val_loader=val_loader,
                                    encoder=encoder, decoder=decoder, D = D, criterion=criterion,
                                    lambd=lambd, convsize=convsize, std=std, device=device)

        # visualize the last batch of validate epoch
        visualize(vis_dir, imgs, pred_vis, None, None, None, enter, esc, length, epoch)

        # Check if there was an improvement
        G_is_best = G_loss < g_best_loss
        G_best_loss = min(G_loss, g_best_loss)
        if not G_is_best:
            epochs_since_G_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_D_improvement))
        else:
            epochs_since_G_improvement = 0

        D_is_best = D_loss < d_best_loss
        D_best_loss = min(D_loss, d_best_loss)
        if not D_is_best:
            epochs_since_D_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_D_improvement))
        else:
            epochs_since_D_improvement = 0
        # Save checkpoint , gan
        # how is it without gan?
        save_checkpoint_gan(save_path, epoch, epochs_since_G_improvement, epochs_since_D_improvement, encoder, decoder, D, encoder_optimizer,
                        decoder_optimizer, D_optimizer, G_loss, G_is_best, D_loss, D_is_best)

if __name__ == '__main__':
    main()
