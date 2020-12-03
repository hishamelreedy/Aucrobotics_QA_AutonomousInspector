#For accessing files
import os
import glob

#For Images
import cv2
import matplotlib.pyplot as plt
import numpy as np

#For checking progress
#from tqdm import tqdm_notebook

import datetime

#PyTorch Packages
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

def get_image(path,transform=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if transform:
        img = transform(img)
    return img

def show_data(rows,cols,is_train=True,transform=False):
    if is_train:
        path = os.path.abspath(os.getcwd()+'\\archive\\dataset\\dataset\\train\\')
    else:
        path = os.path.abspath(os.getcwd()+'\\archive\\dataset\\dataset\\test\\')    
    path = os.path.join(path,'*','*.png')
    img_paths = glob.glob(path)
    np.random.seed(0)
    img_paths = np.random.choice(img_paths,rows*cols)
    fig = plt.figure(figsize=(8,8),dpi=150)
    i = 1
    for r in range(rows):
        for c in range(cols):
            image_path = img_paths[i-1]
            if 'fresh' in image_path.split('\\')[-2]:
                title = 'Fresh'
            else:
                title = 'Rotten'
            ax = fig.add_subplot(rows,cols,i)
            img = get_image(image_path,transform)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title,fontsize=5)
            ax.imshow(img)
            i+=1
    return fig

class FruitsDataset(Dataset):
    def __init__(self,path,classifier_type='Rotten',subset='train',transforms=None):
        self.subset = subset
        if self.subset == 'train':
            self.PATH = os.path.join(path,'train','*','*.png')
        elif self.subset == 'test' and classifier_type=='Rotten':
            self.PATH = os.path.join(path,'test','rottenapples','*.png')
        else:
            self.PATH = os.path.join(path,'test','freshapples','*.png')
        #print(self.PATH)
        self.data = glob.glob(self.PATH)
        self.height = 32
        self.width = 32
        self.labels = [] 
        if classifier_type == 'Rotten':
            classes = ['fresh','rotten']
            for fruit in self.data:
                if classes[0] in fruit.split('\\')[-2]:
                    self.labels.append(0)
                else:
                    self.labels.append(1)
        else:
            classes = ['apple','banana','orange']
            for fruit in self.data:
                if classes[0] in fruit:
                    self.labels.append(0)
                elif classes[1] in fruit:
                    self.labels.append(1)
                else:
                    self.labels.append(2)
        self.transforms = transforms
      
    def __getitem__(self,index):
        img_path = self.data[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.width,self.height))
        label = self.labels[index]
        if self.transforms is not None:
            img_as_tensor = self.transforms(img)
            if self.transforms is not None:
                return(img_as_tensor,label)
            return(img,label)
  
    def __len__(self):
        return(len(self.data))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(16,8,kernel_size=3,padding=1)
        self.fc1 = nn.Linear(8*8*8,32)
        self.fc2 = nn.Linear(32,2)
    def forward(self,x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)),2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)),2)
        out = out.view(-1,8*8*8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
model = Net()
numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
device = torch.device('cpu')
checkpoint = torch.load(os.getcwd()+'\\FreshnessDetector.pt')
model.load_state_dict(checkpoint)
import random
transformations = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.7321, 0.6322, 0.5291),
                                                           (0.3302, 0.3432, 0.3701))
                                      ])
transformations_test = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.7369, 0.6360, 0.5318),
                                                           (0.3281, 0.3417, 0.3704))
                                      ])
badtest = FruitsDataset(os.getcwd()+'\\archive\\dataset\\dataset\\',subset='test',transforms=transformations_test)
goodtest= FruitsDataset(os.getcwd()+'\\archive\\dataset\\dataset\\',subset='test',classifier_type = 'rotten',transforms=transformations_test)
def plotandpredict(name):
    if name=='green':
        #bad apple
        img,label = badtest[random.randint(1,100)];
        fig2=plt.figure();
        plt.imshow(img.permute(1,2,0))
        fig2.savefig('test.png')
        s = nn.Softmax(dim=1)
        out = s(model(img.unsqueeze(0).to(device)))
        if out[0][0]>out[0][1]:
            print('Prediction: fresh')
        else:
            print('Prediction: rotten')
    else:
        #good apple
        img,label = goodtest[random.randint(1,100)];
        plt.figure();
        plt.imshow(img.permute(1,2,0))
        s = nn.Softmax(dim=1)
        out = s(model(img.unsqueeze(0).to(device)))
        if out[0][0]>out[0][1]:
            print('Prediction: fresh')
        else:
            print('Prediction: rotten')

plotandpredict(name='green');