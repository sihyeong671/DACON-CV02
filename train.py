import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from copy import deepcopy

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision.models as models

from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from Modules.Utility import seed_everything, Args, get_data



def train(args: Args):

    seed_everything(args.seed)
    train_df_path, trian_df_label, val_df_path, val_df_label = get_data(args)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--beta', default=1)
    parser.add_argument('--data_path', default='./data/train_repaired.csv')

    args = Args(parser.parse_args())
    train(args)
