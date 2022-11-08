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
import json
import wandb
import Modules.CustomModel
from glob import glob

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")


def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    
    model_preds = []
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            img = data['image'].float().to(device)
            rgb_mean = data['rgb_mean'].float().to(device)
            size = data['size'].float().to(device)
            
            model_pred = model(img, size, rgb_mean)
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
    
    print('Done.')
    return model_preds


def validation(model, criterion, test_loader, device):
    model.eval()
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            img = data['image'].float().to(device)
            label = data['label'].to(device)
            rgb_mean = data['rgb_mean'].to(device)
            size = data['size'].float().to(device)

            model_pred = model(img, size, rgb_mean)
            
            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
        
    val_f1 = competition_metric(true_labels, model_preds)
    return np.mean(val_loss), val_f1
def auto_set_attribute(obj, args:"dict[str, object]"):
    for key in args:
        if hasattr(obj, key):
            setattr(obj, key, args[key])
        else:
            raise Exception("Unknown Attribute -> name : {0}".format(key))
    self = obj
    for attr in self.__dict__:
        if(type(getattr(self, attr)) != str): continue
        if(getattr(self, attr)[0] != '&'):continue
        setattr(self, attr, eval(getattr(self, attr)[1:]))

def auto_check_attribute(obj):
    for attr in obj.__dict__:
        if(getattr(obj, attr) == None or getattr(obj, attr) == 'None'):
            raise Exception("All attributes MUST NOT be None -> name : {0}".format(attr))

class ArgsBase:
    def __init__(self, args) -> None:
        args = vars(args)
        self.batch_size = 32
        self.seed = 999
        self.device = 'cuda'
        self.data_path = './data'
        self.local_path = './local'
        self.config_path = './local'
        self.model_weight_path = './local'
        self.sample_submission_name = 'sample_submission.csv'
        self.train_data_name = 'train_repaired.csv'
        self.test_data_name = 'test.csv'
        self.wandb_entity_name = 'dacon-artist-cv02'
        self.wandb_enable = True

class TrainArgs(ArgsBase):
    def __init__(self, args):
        super(TrainArgs, self).__init__(args)
        self.epochs = 45
        self.lr = 1e-3
        args = vars(args)
        self.beta = 1
        self.scheduler_step = 20
        self.img_size = 380
        self.step_decay = 0.1
        self.wandb_run_name = '&self.model_generator'
        self.wandb_project_name = 'DACON-ArtistClassify'
        self.start_time = '%Y-%m-%d %H.%M.%S'
        self.model_generator = 'None'
        self.save_weight_name = 'Untitled_Weight_Name.tar'
        auto_set_attribute(self, args)
        auto_check_attribute(self)

class TestArgs(ArgsBase):
    def __init__(self, args):
        super(TestArgs, self).__init__(args)
        args = vars(args)
        self.load_weight_name = 'None' #'60epoch_best_EfficientNet_B4_v0'
        auto_set_attribute(self, args)
        auto_check_attribute(self)

def convert_args_to_dict(args:ArgsBase):
    dt = {}
    for attr in args.__dict__:
        dt[attr] = getattr(args, attr)
    return dt

def get_data(args: TrainArgs, sampling: bool = True):
    print("read csv from [", os.path.join(args.data_path, args.train_data_name),']')
    df = pd.read_csv(os.path.join(args.data_path, args.train_data_name))
    le = preprocessing.LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)
    

    if sampling:
        g = df.groupby('artist', group_keys=False)
        train_df_sample = pd.DataFrame()

        for _ in range(15): # 원하는 만큼 sampling
            train_df_sample = pd.concat([train_df_sample, g.apply(lambda x: x.sample(21))])
    else:
        train_df_sample = df
        
    return train_df_sample.img_path.values, train_df_sample.artist.values, # type: ignore


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


def save_model(model, args:TrainArgs, path: str):
    args.model_generator
    torch.save({
        'model_params': model.state_dict(),
        'args': args
    }, path+f'_{args.model_generator}.pth')

def load_model(args:TestArgs) -> "tuple[torch.nn.Module, TrainArgs]":
    ckpt = torch.load(os.path.join(args.model_weight_path, args.load_weight_name), map_location=args.device) 
    print(ckpt['args'].model_generator) # type: ignore
    model = eval('Modules.CustomModel.' + ckpt['args'].model_generator).to(args.device)
    model.load_state_dict(ckpt['model_params'])
    model.eval()
    train_args = ckpt['args']
    return model, train_args


def save_to_csv(args: TestArgs, preds, path: str):

    df = pd.read_csv(os.path.join(args.data_path, args.train_data_name))
    le = preprocessing.LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)
    
    preds = le.inverse_transform(preds)
    submit = pd.read_csv(os.path.join(args.data_path, args.sample_submission_name))

    submit['artist'] = preds

    submit.to_csv(path, index=False)
    return preds

def init_wandb(train_args:TrainArgs, test_args = None):  # type: ignore
    config = convert_args_to_dict(train_args)
    if(test_args != None):
        config.update(convert_args_to_dict(test_args))
    wandb.init(project=train_args.wandb_project_name, entity=train_args.wandb_entity_name, name=train_args.wandb_run_name, config=config)
    wandb.config = convert_args_to_dict(train_args)
    
def get_acc_and_f1(out_a: torch.Tensor, out_b: torch.Tensor, label_a: torch.Tensor, label_b: torch.Tensor, lam: float) -> tuple([int, int]):
    
    target_a_lst = label_a.detach().cpu().numpy().tolist()
    target_b_lst = label_b.detach().cpu().numpy().tolist()

    model_preds_a = out_a.argmax(1).detach().cpu().numpy().tolist()
    model_preds_b = out_b.argmax(1).detach().cpu().numpy().tolist()

    train_f1_a = competition_metric(target_a_lst, model_preds_a) * lam
    train_f1_b = competition_metric(target_b_lst, model_preds_b) * (1. - lam)

    train_acc_a = (label_a==out_a.argmax(1)).sum().item() * lam
    train_acc_b = (label_b==out_b.argmax(1)).sum().item() * (1. - lam)

    train_f1 = train_f1_a + train_f1_b
    train_acc = train_acc_a + train_acc_b

    return train_f1, train_acc

