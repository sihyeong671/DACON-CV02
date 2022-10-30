from tqdm import tqdm
import random
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

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


def saveModel(model, optimizer, args, datetime, path):
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'save_time': datetime,
    }, path)


def loadModel(path: str):
    obj = torch.load(path)
    return obj['model'], obj['optim'], obj['args'], obj['save_time']