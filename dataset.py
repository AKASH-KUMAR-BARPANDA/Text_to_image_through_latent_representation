from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import pandas as pd
import os

""" DATASET CLASS"""
class customdataset(Dataset):
    def __init__(self,imageDir,caption):
        self.imgDir = imageDir
        self.caption = caption
        self.df = pd.read_csv(self.caption)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.iloc[index]["image"]
        caption = self.df.iloc[index]["caption"]
        img_path = os.path.join(self.imgDir,img_name)
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Could not load {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, caption
        

