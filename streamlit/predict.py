import streamlit as st
import torch
import numpy as np
import sys

from utils import transform_img

@st.cache
def load_model() -> EfficientNet_B4:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet_B4(50).to(device)
    ckpt = torch.load('./local/40_best_EfficientNet_B4_EfficientNet_B4(50).pth', map_location=device)
    model.load_state_dict(ckpt['model_params'])
    return model

def get_prediction(model: EfficientNet_B4, image: np.array):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # np.mean(image, axis=(-2, -1))
    tensor = transform_img(image).to(device)
    pred = model(tensor)
    return tensor, pred
    