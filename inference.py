import argparse
import os
import pandas as pd
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn
from torch.utils.data import DataLoader

from Modules.Utility import TestArgs, save_to_csv
from Modules.CustomDataset import CustomDatasetV2
from Modules.CustomModel import EfficientNet_B4

def inference_and_save(args: TestArgs):

    ckpt = torch.load(os.path.join(args.save_model_dir, args.model_name), map_location=args.device)
    model = EfficientNet_B4(50).to(args.device)
    model.load_state_dict(ckpt['model_params'])
    model.eval()

    test_df = pd.read_csv(args.data_path)
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
    
    save_to_csv(args, model_preds, os.path.join(args.save_csv_dir, f'{args.model_name}.csv'))

    print('Done.')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=380)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--train_data_path', default='./data/train.csv')
    parser.add_argument('--test_data_path', default='./data/test.csv')
    parser.add_argument('--save_model_dir', default='./models')
    parser.add_argument('--model_name', default='60epoch_best_EfficientNet_B4_v0')
    parser.add_argument('--save_csv_dir', default='./csv')
    parser.add_argument('--sample_submission_path', default='./data/sample_submission.csv')

    args = TestArgs(parser.parse_args())
    inference_and_save(args)