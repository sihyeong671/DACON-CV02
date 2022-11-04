from tqdm import tqdm
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def validation(model, criterion, test_loader, device):
    model.eval()
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    
    with torch.no_grad():
        for img, label, size in tqdm(test_loader):
            img, label = img.float().to(device), label.to(device)

            size = size.float().to(device)
            model_pred = model(img, size)
            # model_pred = model(img)
            
            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
        
    val_f1 = competition_metric(true_labels, model_preds)
    return np.mean(val_loss), val_f1


class TrainArgs:
    def __init__(self, args):
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.img_size = args.img_size
        self.device = args.device
        self.beta = args.beta
        self.data_path = args.data_path
        self.scheduler_step = args.scheduler_step
        self.save_model_dir = args.save_model_dir
        self.step_decay = args.step_decay

class TestArgs:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.device = args.device
        self.train_data_path = args.train_data_path
        self.test_data_path = args.test_data_path
        self.save_model_dir = args.save_model_dir
        self.model_name = args.model_name
        self.save_csv_dir = args.save_csv_dir
        self.sample_submission_path = args.sample_submission_path
        


def get_data(args: TrainArgs, sampling: bool = True):
    df = pd.read_csv(args.data_path)
    le = preprocessing.LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)
    

    # startify data
    train_df, val_df, _, _ = train_test_split(df, df.img_path.values, test_size=0.2, random_state=args.seed, stratify=df.artist.values)

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


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class F1Loss(nn.Module):
    def __init__(self, classes=50, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


def save_model(model_param, path: str):

    version = 'v' + len(glob(path+'_*'))+1

    torch.save({
        'model_params': model_param
    }, path+f'_{version}.pth')


def save_to_csv(args: TestArgs, preds, path: str):

    df = pd.read_csv(args.train_data_path)
    le = preprocessing.LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)
    preds = le.inverse_transform(preds)
    submit = pd.read_csv(args.sample_submission_path)

    submit['artist'] = preds

    submit.to_csv(path, index=False)