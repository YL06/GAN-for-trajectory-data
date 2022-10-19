#coding:utf-8
import os
import copy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from vis import *
from collections import Counter

class GuiderDataset(Dataset):

    def __init__(self, dir_path, test_size, max_len, min_len=20, target_inter=0.8):


        #float_formatter = "{:12.6f}".format
        float_formatter = "{:.16f}".format
        np.set_printoptions(threshold=1024, linewidth=160, formatter={'float_kind':float_formatter})
        #'C:/Users/info/YJ/Second_stage_gan/train_dataset_res.txt'
        out_fn = 'home/lee/second-stage-GAN/train_dataset_res.txt'
        res_file = open(out_fn, 'w')

        self.dir = dir_path
        self.max_len = max_len # use to padding all sequence to a fixed length
        self.min_len = min_len # use to delete sequence with too many points
        self.target_inter = target_inter
        # deleted / for windows
        self.pic_dir = os.path.join(self.dir,'map_big')
        self.seq_dir = os.path.join(self.dir,'seq')
        self.pic_name = []
        self.data = [] # use to store all seq
        self.train_data = []
        self.test_data = []
        self.trans = torchvision.transforms.ToTensor()

        for image in os.listdir(self.pic_dir):
            image_name = image.rsplit('.')[0]
            self.pic_name.append(image_name)

        # data preprocess:
        # 1. normalize all coordinate according to the first element(x,y,w,h) in each npy file
        # 2. append the image file name to each coordinate sequence
        # 3. concate all sequence in npy file into one
        for k, image_name in enumerate(self.pic_name):
            #print("k", k, "img name", image_name)
            # print("image name", image_name)
            # check if numpy file for the grid exists
            if not os.path.exists(os.path.join(self.seq_dir,image_name+'.npy')):
                continue
            data = np.load(os.path.join(self.seq_dir,image_name+'.npy'),allow_pickle=True)
            print("k:{} image_name:{} size:{} shape:{} ndim:{} ".format(k, image_name, data.size, data.shape, data.ndim), file=res_file)
            #if k in [192, 198 ]:
            #    print("loaded_data:{}".format(data), file=res_file)

            #print("image_name", image_name)
            # numpy array to list
            # data = data.tolist()
            # coordinated already normalized
            #anchor = data[0] # no use for it
            #x,y = anchor[0] # not sure how the authors preprocessed their data for this stage
            #w,h = anchor[1] # we "anchored" it in preprocess
            # added 0 for deleting the "anchor point"
            #data = np.delete(data, 0)  # !!! here delete the anchor point

            # seq is list of list
            maxseq = 0
            for seq in data:
                #print("k:{} maxseq:{} data:{} seq:{} type:{}".format(k, maxseq, data, seq, type(seq)), file=res_file)
                if maxseq > 100:
                    continue

                # check if it's still in numpy
                if isinstance(seq, np.ndarray):
                    seq = seq.tolist()
                    #print("k:{} yet numpy image_name:{}".format(k, image_name), file=res_file)
                    #if k in [10, 11, 13, 65, 381 ]:
                    #    print("seq:{}".format(seq), file=res_file)

                #print("k:{} seq_len:{} seq:{}".format(k, len(seq), seq), file=res_file)

                if len(seq) > self.min_len:
                    print("k:{} too long seq:{}".format(k, image_name), file=res_file)
                    # omit too long
                    continue

                """
                if cal_dis(seq) < 0.2:
                    print("k:{} too short seq:{}".format(k, image_name), file=res_file)
                    # print("to short?", cal_dis(seq))
                    continue # delete seq too short

                if intervals_avg(seq) > 1:
                    print("k:{} too large interval seq:{}".format(k, image_name), file=res_file)
                #    print("check", intervals_avg(seq))
                    continue
                #if seq[-1] == [2,2] or seq[0] == [2,2]:
                #    continue
                #    seq = seq[:-1] # !!! 暂不考虑, 直接删掉终止点，作为一个断掉的轨迹
                #    seq = seq[:-1]
                    # translation: Ignore it for now, just delete the termination point as a broken track

                """

                for i in range(len(seq)):
                    #print("seq[{}]: {}".format(i,seq[i]),  file=res_file)

                    if isinstance(seq[i], tuple):
                        seq[i] = list(seq[i])
                        #print("k:{} tuple to list i={} seq={}".format(k, i, seq[i]), file=res_file)

                    #code for data that hasn't been normalized
                    #seq[i][0] = 2. * (seq[i][0] - x) / w - 1. # rescale to (-1,1)
                    #seq[i][1] = 2. * (seq[i][1] - y) / h - 1.
                    # it seems data has some error.
                    if seq[i][0] < -1. or seq[i][0] > 1.:
                        #print("ERROR k:{} out of X range seq i={} seq={}".format(k, i, seq[i][0]), file=res_file)
                        wrong = seq[i][0]
                        seq[i][0] = round(seq[i][0])
                        #print("FIXED rounded to seq({}):{} to {}".format(i, wrong, seq[i][0]), file=res_file)
                        #seq[i] = [0., 0.]
                    if seq[i][1] < -1. or seq[i][1] > 1.:
                        #print("ERROR k:{} out of Y range seq i={} seq={}".format(k, i, seq[i][1]), file=res_file)
                        #seq[i] = [0., 0.]
                        wrong = seq[i][1]
                        seq[i][1] = round(seq[i][1])
                        #print("FIXED rounded to seq=({}):{} to {}".format(i, wrong, seq[i][1]), file=res_file)

                    #diff = seq[i][1] + 1
                    #if seq[i][1] > highest_y and seq[i][1] < 1.:
                    #    print("FOUND k:{} i={} Y_high={} seq={}".format(k, i, highest_y, seq[i][1]), file=res_file)


                    # hided in comments, what is this
                    # TODO: look this up in the paper
                    #seq[i][0] = 2 * seq[i][0] - 1
                    #seq[i][1] = 2 * seq[i][1] - 1
                # print("how is seq looking?", seq)
                # use np append? but we dont want to, we want a list
                # seq = np.append(seq, image_name)
                seq.append(image_name) # append cooresponding map image name to each sequence
                # print("seq with image", seq, type(seq))
                # added data append

                if k < 20:
                    self.test_data.append(seq)
                else:
                    self.train_data.append(seq)
                maxseq += 1
                #self.data.append(seq) # seq is a list of list

        #self.train_data, self.test_data = train_test_split(self.data, test_size=test_size,random_state=0)
        print("="*50, file=res_file)
        print("Data Preprocess Done!")
        print("size", len(self.train_data))#
        print("vali", len(self.test_data))
        print("Dataset size:{}, train:{}, val:{}".
              format(len(self.data),len(self.train_data),len(self.test_data)), file=res_file)
        print("="*50, file=res_file)
        print("Done Dataset!")

        res_file.close()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq = copy.deepcopy(self.data[item]) # a list
        seq_len = len(seq) - 1 # except the last filename element

        trans = transforms.Compose(
            [transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        image_name = seq[-1]
        seq = seq[:-1]
        image_path = os.path.join(self.pic_dir,image_name+'.png')
        image = Image.open(image_path)
        tensor = trans(image) # (C,W,H)

        enter_point = torch.tensor(seq[0],dtype=torch.float) # dim = 2
        esc_point = torch.tensor(seq[-1],dtype=torch.float) # dim = 2

        # <----- data preprocess ------->
        # pay attetion! seq_inv differs in the following two situation
        if seq_len <= self.max_len:
            seq_inv = seq[::-1]
            seq += [[0., 0.] for _ in range(self.max_len - seq_len)]
            seq_inv += [[0., 0.] for _ in range(self.max_len - seq_len)]
            seq_inv = torch.tensor(seq_inv, dtype=torch.float)
        elif seq_len > self.max_len:
            # systematically sample a subset for long sequence
            dis = int(seq_len / self.max_len)
            ind = [dis*i for i in range(self.max_len)]
            seq = [seq[i] for i in ind]
            seq_inv = torch.tensor(seq[::-1], dtype=torch.float) # (max_len,2) inverse orde
            seq_len = self.max_len # be careful!
        # <----------------------------->

        seq = torch.tensor(seq ,dtype=torch.float) # (max_len, 2)
        seq_len = torch.tensor(seq_len,dtype=torch.long).unsqueeze(0)

        return {'name':image_name, 'image':tensor, 'seq':seq, 'seq_inv':seq_inv, 'enter':enter_point, 'esc':esc_point, 'len':seq_len}

    def train_set(self):
        '''call this method to switch to train mode'''
        self.data = self.train_data
        return copy.deepcopy(self)

    def test_set(self):
        '''call this method to switch to test mode'''
        self.data = self.test_data
        return copy.deepcopy(self)

def intervals_avg(seq):
    '''used to calculate the average intervals of a certain sequence'''
    #seq = seq.tolist()
    seq_ = np.array(seq[1:])
    seq = np.array(seq[:-1])
    intervals = np.sqrt(np.power(seq-seq_,2).sum(axis=1)).mean()
    return intervals


def cal_dis(seq):
    '''calculate the distance between seq[0] and seq[-1]'''
    x1 = seq[0][0]
    y1 = seq[0][1]
    x2 = seq[-1][0]
    y2 = seq[-1][1]
    dis = np.sqrt((y2 - y1)**2 + (x2 - x1)**2).item()
    return dis


# debug
if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    data_path = 'D:/BA/Two-Stage-Gan-in-trajectory-generation/Second_stage_gan/data/'
    dataset = GuiderDataset(data_path,0.2,max_len=20)
    print(dataset.data)
    data = dataset.data[10]
    cache = []
    for i in range(len(dataset.data)):
        cache.append(len(dataset.data[i]))
    summary = Counter(cache)
    print(summary)

    ## visualize dataset
    output_path = 'output'
    train_loader = DataLoader(dataset.train_set(), batch_size=8, shuffle=False)
    for i,data in enumerate(train_loader):
        #save_dir = os.path.join(output_path,str(i))
        save_dir = output_path
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        imgs = data['image'] # (b,c,w,h) b, batchsize
        print("img", np.shape(imgs))
        seq = data['seq'] # (b,max_len,2)
        print("seq", np.shape(seq))
        enter = data['enter'] # (b,2)
        print("enter", np.shape(enter))
        esc = data['esc'] # (b,4) # onehot indicate 4 direction
        print("esc", np.shape(esc))
        length = data['len'] # (b,1) it seem to be a 1D CPU int64 tensor when use pack_padded_sequence below
        #print(imgs)
        for k in range((len(data['image']))):
            print("k", k)
            img = imgs[k].numpy()
            c,w,h = img.shape
            print(c,w,h)
            img[0] = img[0]*0.229+0.485
            img[1] = img[1]*0.224+0.456
            img[2] = img[2]*0.225+0.406
            img = img * 255
            img.astype('int')
            img = img.transpose(1,2,0) # chw -> hwc
            img = img[...,::-1] #rgb --> bgr
            #print(enter[k],esc[k])
            img = cv.copyMakeBorder(img, 5, 5, 5, 5, cv.BORDER_CONSTANT,value=[225,225,225])
            for j in range(length[k]):
                m, n = int(h/2+seq[k][j,1]*(h/2-1)),int(h/2+seq[k][j,0]*(h/2-1))
                if seq[k][j,1] == 3:
                    break
                img[-m-3:-m+3,n-3:n+3,:] = np.zeros_like(img[-m-3:-m+3,n-3:n+3,:])
                #print(img[:,int(seq[k][j,1]*h)+256,256+int(seq[k][j,0]*h)])

            print(enter[k], enter[k][1], -int(h/2+5+enter[k][1]*(h/2-1))-3,-int(h/2+5+enter[k][1]*(h/2-1))+3, int(h/2+5+enter[k][0]*(h/2-1))-3, int(h/2+5+enter[k][0]*(h/2-1))+3 )
            # 红色是入点 red is the in point
            img[-int(h/2+5+enter[k][1]*(h/2-1))-3:-int(h/2+5+enter[k][1]*(h/2-1))+3,int(h/2+5+enter[k][0]*(h/2-1))-3:int(h/2+5+enter[k][0]*(h/2-1))+3,0] = 0
            img[-int(h/2+5+enter[k][1]*(h/2-1))-3:-int(h/2+5+enter[k][1]*(h/2-1))+3,int(h/2+5+enter[k][0]*(h/2-1))-3:int(h/2+5+enter[k][0]*(h/2-1))+3,1] = 0
            img[-int(h/2+5+enter[k][1]*(h/2-1))-3:-int(h/2+5+enter[k][1]*(h/2-1))+3,int(h/2+5+enter[k][0]*(h/2-1))-3:int(h/2+5+enter[k][0]*(h/2-1))+3,2] = 200


            # img[-200:-100, 0: h, 0] = 200
            #蓝色 出点方向 blue out point direction
            # if esc[k][0] == 1:
            #     img[-5:,:,0] = 200
            #    img[-5:,:,1] = 0
            #    img[-5:,:,2] = 0
            #elif esc[k][1] == 1:
            #    img[:,:5,0] = 200
            #    img[:,:5,1] = 0
            #    img[:,:5,2] = 0
            #elif esc[k][2] == 1:
            #   img[:5,:,0] = 200
            #    img[:5,:,1] = 0
            #    img[:5,:,2] = 0
            #elif esc[k][3] == 1:
            #    img[:,-5:,0] = 200
            #    img[:,-5:,1] = 0
            #    img[:,-5:,2] = 0
            print(str(i*32+k)+'.png')
            cv.imwrite(os.path.join(save_dir,str(i*32+k)+'.png'),img)
