from turtle import width
import cv2 as cv2
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
            label = None
        return {'image' : img, 'height' : h, 'width' : w, 'channel' : c, 'label' : label}
    
    def __len__(self):
        return len(self.img_paths)