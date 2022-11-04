import cv2
import torch
import os
import albumentations as A
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, img_paths, labels, transforms):
        self.data_path = data_path
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.data_path + self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape        
        
        if(self.transforms is not None):
            img = self.transforms(image=img, width=w, height=h, depth=c)['image']

        if(self.labels is not None):
            label = self.labels[index]
        else:
            label = 'None'
        return {'image' : img, 'height' : h, 'width' : w, 'channel' : c, 'label' : label}
    
    def __len__(self):
        return len(self.img_paths)


class CustomDatasetV2(Dataset):
    def __init__(self, img_paths, labels, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_path = os.path.join('./data', img_path) # 본인 데이터 폴더 위치 확인 필요
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 이미지 길이 1 / 2

        if self.labels is not None: # train
          h, w, _ = image.shape
          size = torch.tensor([h, w])
          l = min(h//2, w//2)
          inner_transform = A.Compose([A.RandomCrop(width=l, height=l)])
        else: # test
          h, w, _ = image.shape
          size = torch.tensor([2*h, 2*w])
        
        if self.transforms is not None:
            if self.labels is not None: # train
              image = inner_transform(image=image)['image']
            image = self.transforms(image=image)['image']
        
        if self.labels is not None:
            label = self.labels[index]
            return image, label, size
        else:
            return image, size
    
    def __len__(self):
        return len(self.img_paths)