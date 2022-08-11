'''
    Creating a dataset loader for a series of images and csv file

    2022 Benjamin Kellenberger
    modified by Natalie Imirzian, 2022
'''

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


class SizeDataset(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            Resize((cfg['image_size'])),        # For now, we just resize the images to the same dimensions...
            ToTensor()                          # ...and convert them to torch.Tensor.
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
            imgFileName = row['filename']
            labelIndex = row['cat']
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
        image_path = os.path.join(self.data_root, 'train', image_name) if self.split=='train' else os.path.join(self.data_root, 'val', image_name)
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)

        return img_tensor, label