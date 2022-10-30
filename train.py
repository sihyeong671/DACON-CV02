import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import datetime
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models
from sklearn import preprocessing
sys.path.append('./') # import를 위해 경로추가
from models import *
from Modules import Utility as U
from Modules import CustomDataset 
from Modules import SmartPad

def validation(model, criterion, test_loader, device):
    model.eval()
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    
    with torch.no_grad():
        for item in tqdm(test_loader):
            img = item['image']
            label = item['label']
            h = item['height']
            w = item['width']
            img, label = img.float().to(device), label.to(device)
            model_pred = model(img)
            
            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
        
    val_f1 = U.competition_metric(true_labels, model_preds)
    return np.mean(val_loss), val_f1

def train(model, optimizer, train_loader, test_loader, scheduler, args, datetime, path_model_weight) -> None:
    model.to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    
    best_score = 0
    
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = []
        for item in tqdm(train_loader):
            img = item['image']
            label = item['label']
            h = item['height']
            w = item['width']
            img, label = img.float().to(args.device), label.to(args.device)
            optimizer.zero_grad()

            model_pred = model(img)
            
            loss = criterion(model_pred, label)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        tr_loss = np.mean(train_loss)
            
        val_loss, val_score = validation(model, criterion, test_loader, args.device)
            
        print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 Score : [{val_score:.5f}]')
        
        if scheduler is not None:
            scheduler.step()
            
        if best_score < val_score:
            best_score = val_score
            print(
                f' * New Best Model -> Epoch [{epoch}] / best_score : [{best_score:.5f}]')
            U.saveModel(model=model, optimizer=optimizer,
                        args=args, datetime=datetime, path=path_model_weight)
            print(" -> The model has been saved at " + path_model_weight)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="params")
    
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--step_gamma', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--img_size', type=int, default=244)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--ref_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--csv_name', type=str, default='train.csv')
    parser.add_argument('--save_path', type=str,
                        default="models/checkpoint/")
    parser.add_argument('--save_name', type=str,
                        default="weights.tar")
    parser.add_argument('--target_model', type=str,
                        default="ResNextV0()")
    args = parser.parse_args()
    print(args)
    target_model = eval(args.target_model)
    U.seed_everything(args.seed)
    df = pd.read_csv(os.path.join(args.ref_path, args.data_path, args.csv_name))
    le = preprocessing.LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)
    train_df, val_df, _, _ = U.train_test_split(df, df['artist'].values, test_size=0.2, random_state=args.seed)
    train_img_paths, train_labels = U.get_data(train_df)
    val_img_paths, val_labels = U.get_data(val_df)
    # random flip
    train_transform = A.Compose([
                                SmartPad(),
                                # A.Resize(args.img_size, args.img_size),
                                A.RandomResizedCrop(height = args.img_size, 
                                                    width = args.img_size,
                                                    scale = (0.5,1)),
                                A.HorizontalFlip(),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    test_transform = A.Compose([
                                SmartPad(),
                                A.Resize(args.img_size,args.img_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])
    train_dataset = CustomDataset(args.data_path, train_img_paths, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0)

    val_dataset = CustomDataset(args.data_path, val_img_paths, val_labels, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = target_model.to(args.device)
    model.eval()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # scheduler = None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)

    # start time
    dt = datetime.datetime.now()

    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=val_loader,
        scheduler=scheduler,
        args=args,
        datetime=dt,
        path_model_weight=os.path.join(args.ref_path, args.save_path, args.save_name)
    )
    



    
