'''
    Creating a dataset loader for a series of images and csv file

    2022 Benjamin Kellenberger
    modified by Natalie Imirzian, 2022
'''

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
#from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import torch
#import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
torch.manual_seed(42)



class SizeDataset(Dataset):

    def __init__(self, cfg, split='train', transform=None):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([ 
                A.ToFloat(max_value=255.0),             
                ToTensorV2(),
                torch.Tensor.double()
            ])  
        
        # index data into list
        self.data = []

        # load annotation file
        annoPath = os.path.join(
            self.data_root,
            'train.'+cfg['experiment']+'.csv' if self.split=='train' else "val."+cfg['experiment']+".csv"
        )

        meta = pd.read_csv(annoPath)
        meta.reset_index()

        for index, row in meta.iterrows():   
            # append image-label tuple to data
            imgFileName = row['filename']
            labelIndex = row['class']
            self.data.append([str(imgFileName), labelIndex])

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]             

        # load image
        image_path = os.path.join(self.data_root, 'images', image_name)
        img = np.array(Image.open(image_path)).astype(np.uint8)     # the ".convert" makes sure we always get three bands in Red, Green, Blue order
        

        img_tensor = self.transform(image=img)
        img_tensor = img_tensor["image"]

        return img_tensor, label



class SimpleDataset(Dataset):

    def __init__(self, cfg, split='train', upsample=False):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        self.upsample = upsample 
        self.transform = Compose([              
            Resize((cfg['image_size'])), 
            ToTensor()                         
        ])
        
        # index data into list
        self.data = []

        # load annotation file
  
        annoPath = os.path.join(
            self.data_root,
            'train_ant_size.csv' if self.split=='train' else "val_ant_size.csv"
        )
    
        meta = pd.read_csv(annoPath)
        meta.reset_index()

        for index, row in meta.iterrows():   
            # append image-label tuple to data
            if row['class'] == 0:
                imgFileName = row['filename']
                self.data.append([str(imgFileName), 0])
            if row['class'] == 4:
                imgFileName = row['filename']
                self.data.append([str(imgFileName), 1])

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]             

        # load image
        image_path = os.path.join(self.data_root, 'images', image_name)
        img = np.array(Image.open(image_path).convert('RGB')).astype(np.uint8)    # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(image=img)["image"]

        return img_tensor, label



class Transform():
    def __init__(self, cfg):

        self.transform = A.Compose([
            A.Rotate(-360, 360),
            A.Flip(cfg['flip_prob']),
            A.Sharpen(alpha=(0.2, 0.5), 
                      lightness=(0.5, 1.0), 
                      p=cfg['sharp_prob']),
            ToTensorV2()])
        

    def __call__(self, img):
        return self.transform(img)


