import os
from glob import glob
import rasterio
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from utils import train_val_files


class Sen1flood11Dst(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        
        image_path = self.df.iloc[idx]['source']
        label_path = self.df.iloc[idx]['label']

        
        if 'c2smsfloods_v1_source_s1' in image_path:
            s1_path = glob(os.path.join(image_path, '*.tif'))
            lbl_paths = glob(os.path.join(label_path, '*.tif'))
            
            vv_path, vh_path = s1_path[0], s1_path[1]
            
            VV = np.nan_to_num(rasterio.open(vv_path).read())[0]
            VH = np.nan_to_num(rasterio.open(vh_path).read())[0]
            
            label = rasterio.open(lbl_paths[0]).read()
            
        else:
            label = rasterio.open(label_path).read()  #(1, 512, 512)
            source = np.nan_to_num(rasterio.open(image_path).read())  #(2, 512, 512)
            VV, VH = source[0], source[1]  #(512, 512)
        
        
        label[label == -1] = 0
        label[label == 255] = 0
        
        VV = (np.clip(VV, -50, 1) + 50) / 51
        VH = (np.clip(VH, -50, 1) + 50) / 51
        
        VV = (VV - 0.6851) / 0.0820
        VH = (VH - 0.6851) / 0.1102
        
        RFCC = np.stack((VV, VH, VH - VV), axis=2)

        if self.transform is not None:
            data = self.transform(image=RFCC, mask=label[0]) #RFCC.transpose((2, 0, 1))
            
            img = torch.tensor(data['image'].transpose((2, 0, 1)), dtype=torch.float32)
            lbl =  torch.tensor(data['mask'][None], dtype=torch.float32)
            
        else:
            img = torch.tensor(RFCC.transpose((2, 0, 1)), dtype=torch.float32) #RFCC.transpose((2, 0, 1))
            lbl = torch.tensor(label, dtype=torch.float32)
            
        return img, lbl


class FloodDataModule(LightningDataModule):
    def __init__(self, args, transformations):
        super().__init__()
        self.args = args
        self.transformations = transformations
    
    
    def setup(self, stage=None):
        train_df, val_df, test_df = train_val_files(self.args)

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        self.train_dataset = Sen1flood11Dst(train_df, transform=self.transformations) 
        self.valid_dataset = Sen1flood11Dst(val_df, transform=None)
        self.test_dataset = Sen1flood11Dst(test_df, transform=None)
        
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.args.batch_size, 
                          num_workers=self.args.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)
        


    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.args.batch_size, 
                          num_workers=self.args.num_workers,
                          pin_memory=False,
                          drop_last=False)
    
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.args.batch_size, 
                          num_workers=self.args.num_workers,
                          pin_memory=False,
                          drop_last=False)
