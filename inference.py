import argparse
import os
import pandas as pd
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Modules import *
import wandb

def inference_and_save(model, train_args: TrainArgs, test_args: TestArgs):
    test_df = pd.read_csv(os.path.join(test_args.data_path, test_args.test_data_name))
    test_img_path = test_df.img_path

    test_transform = A.Compose([
                            A.Resize(train_args.img_size,train_args.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

    test_dataset = CustomDatasetV2(test_img_path, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=test_args.batch_size, shuffle=False, num_workers=2)

    model_preds = []
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            img = data['image'].float().to(test_args.device)
            # label = data['label'].to(test_args.device)
            rgb_mean = data['rgb_mean'].float().to(test_args.device)
            size = data['size'].float().to(test_args.device)
            img = img.float().to(test_args.device)
            size = size.float().to(test_args.device)

            model_pred = F.softmax(model(img, size, rgb_mean).detach().cpu(), dim=1).numpy().tolist()
            model_preds += model_pred
            
    submit = pd.read_csv(os.path.join(test_args.data_path, test_args.sample_submission_name))
    submit['artist'] = model_preds
    submit.to_csv(os.path.join(test_args.local_path, f'{test_args.load_weight_name}_submission.csv'), index=False)
        
    print('Done.')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--load_weight_name', default='40_best_EfficientNet_B4_non_size_rgb_EfficientNet_B4_non_size_rgb(50).pth')

    test_args = TestArgs(parser.parse_args())
    args_dict = convert_args_to_dict(test_args)
    print('***** echo args *****')
    for k in args_dict:
        print(' - ', k, ':', args_dict[k])
    print('*********************')
    model, train_args = load_model(test_args)
    print('> ***** echo loaded train_args *****')
    args_dict = convert_args_to_dict(train_args)
    for k in args_dict:
        print('>  - ', k, ':', args_dict[k])
    print('> *********************')
    inference_and_save(model, train_args, test_args)