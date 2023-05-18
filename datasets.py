import torch
import numpy as np
from matplotlib import pyplot as plt
#import matplotlib as plt
import os
from PIL import Image
from torchvision import transforms
import cv2

class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, training=True, transform=None):
        if training==True:
            f = open('../data/MNIST/raw/train-images-idx3-ubyte', 'rb')
            xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))
            f.close()
            f = open('../data/MNIST/raw/train-labels-idx1-ubyte', 'rb')
            ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))
            f.close()
        else:
            f = open('../data/MNIST/raw/t10k-images-idx3-ubyte', 'rb')
            xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))
            f.close()
            f = open('../data/MNIST/raw/t10k-labels-idx1-ubyte', 'rb')
            ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))
            f.close()
        xs = np.reshape(xs, (-1, 28, 28, 1)).astype(np.float32)
        ys = ys.astype(np.int)
        self.x_data = xs
        self.y_data = ys
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = Image.fromarray(self.x_data[idx].reshape(28, 28))
        y = torch.tensor(np.array(self.y_data[idx]))
        if self.transform:
            x = self.transform(x)
        x = transforms.ToTensor()(np.array(x)/255)
        return x, y

class MnistDataset2(torch.utils.data.Dataset):
    def __init__(self, training=True,testing=True,patchsize=180, transform=None):
        if training==True:
            fh=open("/home/ihclserver/Desktop/deeplearning_homework3_sisheng/images/train.txt","r",encoding="utf-8")
            lines=fh.readlines()
            data=[]
            label=[]
            for line in lines:
                line=line.strip("\n")
                line=line.strip()
                words=line.split()
                imgs_path=words[0]
                labels=words[1]
                label.append(labels)
                data.append(imgs_path)
        elif testing==True:
            fh=open("/home/ihclserver/Desktop/deeplearning_homework3_sisheng/images/test.txt","r",encoding="utf-8")
            lines=fh.readlines()
            data=[]
            label=[]
            for line in lines:
                line=line.strip("\n")
                line=line.strip()
                words=line.split()
                imgs_path=words[0]
                labels=words[1]
                label.append(labels)
                data.append(imgs_path)
        else :
            fh=open("/home/ihclserver/Desktop/deeplearning_homework3_sisheng/images/val.txt","r",encoding="utf-8")
            lines=fh.readlines()
            data=[]
            label=[]
            for line in lines:
                line=line.strip("\n")
                line=line.strip()
                words=line.split()
                imgs_path=words[0]
                labels=words[1]
                label.append(labels)
                data.append(imgs_path)
                

       
        self.x_data = data
        self.y_data = label
        self.size=patchsize
        

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image_dir="/home/ihclserver/Desktop/deeplearning_homework3_sisheng/"
        img_path=os.path.join(image_dir,self.x_data[idx])
        image=plt.imread(img_path)
        if image.ndim==2:
            image=cv2.cvtColor(image,cv2.COLOR_BAYER_BG2BGR)
        resize=transforms.CenterCrop(size=self.size)
        image_tensor=torch.from_numpy(image).to_dense().permute(2,0,1)
        x=resize(image_tensor)
        cla=int(self.y_data[idx])
        y = torch.tensor(np.array(cla))
        return x, y
