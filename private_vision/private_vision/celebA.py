import os
import glob
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd



class private_dataset(Dataset):
    
    def __init__(self, image_loc, label_loc, transform):
        
        filenames = []
        for root, dirs, files in os.walk(image_loc):
            for file in files:
                if file.endswith('.jpg') == True or file.endswith('.png') == True :
                    filenames.append(file)
        
        self.full_filenames = glob.glob(image_loc+'*/*/*.*')
        
        label_df = pd.read_csv(label_loc)
        label_df.set_index("filename", inplace = True)
        self.labels = [label_df.loc[filename].values[0] for filename in filenames]
        self.transform = transform
        
        
    def __len__(self):
        return len(self.full_filenames)
    
    
    def __getitem__(self,idx):
        image = Image.open(self.full_filenames[idx]) 
        image = image.convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]

    def transform(self,crop_size, re_size):
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
        proc = []
        proc.append(transforms.ToTensor())
        proc.append(transforms.ToPILImage())
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.RandomHorizontalFlip())
        proc.append(transforms.ToTensor())
        return transforms.Compose(proc)
