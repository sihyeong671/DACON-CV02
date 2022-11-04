from tqdm import tqdm
import random
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
from glob import glob

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")

def get_data(df, infer=False):
    if infer:
        return df['img_path'].values
    return df['img_path'].values, df['artist'].values

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    
    model_preds = []
    
    with torch.no_grad():
        for item in tqdm(iter(test_loader)):
            img = item['image'].float().to(device)
            h = item['height']
            w = item['width']
            
            model_pred = model(img)
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
    
    print('Done.')
    return model_preds


class Args:
    def __init__(self, args):
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.img_size = args.img_size
        self.device = args.device
        self.beta = args.beta
        self.data_path = args.data_path

def get_data(args: Args, sampling: bool = True):
    df = pd.read_csv(args.data_path)
    le = preprocessing.LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)
    
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    # startify data
    g = df.groupby('artist', group_keys=False)
    for name, p in g:
        train, val, _, _ = train_test_split(p, p.img_path.values, test_size=0.2, random_state=args.seed)
        train_df = train_df.concat((train_df, train))
        val_df = val_df.concat((val_df, val))

    if sampling:
        g = df.groupby('artist', group_keys=False)
        train_df_sample = pd.DataFrame()
        val_df_sample = pd.DataFrame()

        for _ in range(15): # 원하는 만큼 sampling
            train_df_sample = pd.concat([train_df_sample, g.apply(lambda x: x.sample(16))])
            val_df_sample = pd.concat([val_df_sample, g.apply(lambda x: x.sample(5))])
    else:
        train_df_sample = train_df
        val_df_sample = val_df
        
    return train_df_sample.img_path.values, train_df_sample.artist.values, val_df_sample.img_path.values, val_df_sample.artist.values