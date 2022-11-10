import albumentations as A
import numpy as np
import streamlit as st
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Modules import EfficientNet_B4
import os

@st.cache
def load_model() -> EfficientNet_B4:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet_B4(50).to(device)
    ckpt = torch.load('./local/EfficientNet_B4(50).pth', map_location=device)
    model.load_state_dict(ckpt['model_params'])
    return model


def get_prediction(model: EfficientNet_B4, image: np.array):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h, w, _ = image.shape
    size = torch.tensor([h, w], dtype=torch.float32).unsqueeze(0).to(device)
    rgb = torch.tensor(np.mean(image, axis=(0, 1)), dtype=torch.float32, device=device).unsqueeze(0)
    tensor = transform_img(image).to(device)
    out = model(tensor, size, rgb)


    df = pd.read_csv(os.path.join('./data', 'train_repaired.csv'))
    le = LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)

    pred = out.argmax(1).detach().cpu().numpy().tolist()
    
    pred = le.inverse_transform(pred)
    return tensor, pred


def transform_img(image: np.array):
    transform = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.transforms.ToTensorV2()
    ])

    return transform(image=image)['image'].unsqueeze(0)
    