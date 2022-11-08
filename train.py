import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from copy import deepcopy

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from Modules.Utility import seed_everything, TrainArgs, get_data, rand_bbox, validation, F1Loss, save_model
from Modules.CustomDataset import CustomDatasetV2
from Modules.CustomModel import EfficientNet_B4


def train_and_save(args: TrainArgs):

    seed_everything(args.seed)

    train_df_path, train_df_label, val_df_path, val_df_label = get_data(args)

    train_transform = A.Compose([
                            A.VerticalFlip(),
                            A.HorizontalFlip(),
                            A.Resize(args.img_size,args.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            # A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

    test_transform = A.Compose([
                            A.Resize(args.img_size,args.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            # A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

    train_dataset = CustomDatasetV2(train_df_path, train_df_label, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=2)

    val_dataset = CustomDatasetV2(val_df_path, val_df_label, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = EfficientNet_B4(50).to(args.device) # num_classes

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(params = model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.step_decay)

    best_score = 0
    best_loss = np.inf
    best_model = None

    es_check = 0
    es_limit = 7
    
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = []
        for data in tqdm(train_loader):
            img = data['image'].float().to(args.device)
            label = data['label'].to(args.device)
            rgb_mean = data['rgb_mean'].float().to(args.device)
            size = data['size'].float().to(args.device)

            # cutmix
            r = np.random.rand(1)
            if r < 0.5:
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(img.size()[0]).to(args.device)

                target_a = label
                target_b = label[rand_index]
                size_a = size
                size_b = size[rand_index]
                rgb_mean_a = rgb_mean
                rgb_mean_b = rgb_mean[rand_index]

                bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
                img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
                # compute output
                size = size_a * lam + size_b * (1. - lam)
                rgb_mean = rgb_mean_a * lam + rgb_mean_b * (1. - lam)
                outs_a = model(img, size_a, rgb_mean_a)
                outs_b = model(img, size_b, rgb_mean_b)
                loss = criterion(outs_a, target_a) * lam + criterion(outs_b, target_b) * (1. - lam)
            else:
                outs = model(img, size, rgb_mean)
                loss = criterion(outs, label)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        tr_loss = np.mean(train_loss)
            
        val_loss, val_score = validation(model, criterion, val_loader, args.device)
            
        print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 Score : [{val_score:.5f}]')
        
        if scheduler is not None:
            scheduler.step()
            
        if best_score < val_score:
            best_model = deepcopy(model.state_dict())
            best_score = val_score
            es_check = 0
        else:
            es_check += 1

        # early stopping
        if es_check > es_limit:
            break

    
    save_model(best_model, os.path.join(args.save_model_dir, f'{args.epochs}_best_{model.__class__.__name__}'))
        
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--scheduler_step', default=30)
    parser.add_argument('--step_decay', default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--img_size', type=int, default=380)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--beta', default=1)
    parser.add_argument('--data_path', default='./data/train_repaired.csv')
    parser.add_argument('--save_model_dir', default='./models')

    args = TrainArgs(parser.parse_args())
    train_and_save(args)
