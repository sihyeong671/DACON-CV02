import argparse
import os
import pandas as pd
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn
from torch.utils.data import DataLoader

from Modules.Utility import TrainArgs,TestArgs, save_to_csv, load_model
from Modules.CustomDataset import CustomDatasetV2
from Modules.CustomModel import EfficientNet_B4

def inference_and_save(args: TestArgs):
    model, trainArgs = load_model(args)
    test_df = pd.read_csv(os.path.join(args.data_path, args.test_data_name))
    test_img_path = test_df.img_path

    test_transform = A.Compose([
                            A.Resize(args.img_size,args.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

    test_dataset = CustomDatasetV2(test_img_path, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model_preds = []
    
    with torch.no_grad():
        for img, size in tqdm(iter(test_loader)):
            img = img.float().to(args.device)
            size = size.float().to(args.device)
            model_pred = model(img, size)
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
    
    save_to_csv(args, model_preds, os.path.join(args.local_path, f'{args.load_weight_name}_submission.csv'))

    print('Done.')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--load_weight_name', default='60epoch_best_EfficientNet_B4_v0.tar')

    args = TestArgs(parser.parse_args())
    inference_and_save(args)