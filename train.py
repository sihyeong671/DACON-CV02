import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime as dt
from tqdm import tqdm


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from Modules import *

import wandb


def train_and_save(args: TrainArgs):

    seed_everything(args.seed)

    train_df_path, train_df_label = get_data(args)

    train_transform = A.Compose([
                            A.VerticalFlip(),
                            A.HorizontalFlip(),
                            A.Resize(args.img_size,args.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])


    train_dataset = CustomDatasetV2(train_df_path, train_df_label, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=2)

    model = eval(args.model_generator).to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(params = model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.step_decay)

    best_score = 0
    NUM_ACCUM = 2
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = []
        train_f1 = []
        train_acc = []
        for idx, data in enumerate(tqdm(train_loader)):
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
                out_a = model(img, size_a, rgb_mean_a)
                out_b = model(img, size_b, rgb_mean_b)

                train_f1_item, train_acc_item = get_acc_and_f1(out_a, out_b, target_a, target_b, lam)
                
                train_f1.append(train_f1_item)
                train_acc.append(train_acc_item)

                loss = criterion(out_a, target_a) * lam + criterion(out_b, target_b) * (1. - lam)
            else:
                outs = model(img, size, rgb_mean)
                model_preds = outs.argmax(1).detach().cpu().numpy().tolist()
                label_lst = label.detach().cpu().numpy().tolist()
                train_f1_item = competition_metric(label_lst, model_preds)
                train_f1.append(train_f1_item)
                loss = criterion(outs, label)

            loss.backward()

            if idx % NUM_ACCUM == 0:
                optimizer.zero_grad()
                optimizer.step()

            train_loss.append(loss.item())

        tr_loss = np.mean(train_loss)
        train_f1_score = np.mean(train_f1)
        train_acc_score = np.mean(train_acc)
            
            
        print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Train Acc : [{train_acc_score:.5f}] Train F1 Score : [{train_f1_score:.5f}]')
        wandb.log({"Train Loss": tr_loss, "Train Acc":train_acc_score, "Train F1 Score":train_f1_score})
        if scheduler is not None:
            scheduler.step()

        if best_score < train_f1_score:
            best_score = train_f1_score
            args.save_weight_name = f'{args.epochs}_best_{model.__class__.__name__}'
            save_model(model, args, os.path.join(args.model_weight_path, args.save_weight_name))
        
    
if __name__ == '__main__':
    now = dt.datetime.now().strftime('%Y-%m-%d %H.%M.%S')
    print('The training started at ', now)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=653) # 
    parser.add_argument('--start_time', type=str, default=now)
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--scheduler_step', default=30)
    parser.add_argument('--step_decay', default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=380)
    parser.add_argument('--beta', default=1)
    parser.add_argument('--model_generator', default="ResNeXt101(50)")
    parser.add_argument('--wandb_enable', default=True)
    args = TrainArgs(parser.parse_args())
    args_dict = convert_args_to_dict(args)

    print('***** echo args *****')
    for k in args_dict : print(' - ', k, ':', args_dict[k])
    print('*********************')

    if(args.wandb_enable) : init_wandb(args)

    train_and_save(args)
