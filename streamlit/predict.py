import yaml
import torch
import streamlit as st
from typing import Tuple

import io
from PIL import Image
import sys
sys.path.append('../')
from Modules import *
from utils import transform_image

@st.cache
def load_model() -> ResNeXt50:
    with open("artist.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load(config['model_path'], map_location=device)
    model = ResNeXt50(50).to(device)
    
    model.load_state_dict(ckpt['model_params'])
    
    return model



def get_prediction(model:ResNeXt50, image_bytes ) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(io.BytesIO(image_bytes))
    
    h, w = image.size
    
    image_rgb = np.array(image)
    image_rgb.shape = (w*h, 3)
    image_rgb = np.average(image_rgb, axis=0)
    image_rgb = np.expand_dims(image_rgb, axis=0)
    rgb_mean = torch.tensor(image_rgb).float().to(device)
    
    
    h *= 2
    w *= 2
    size = torch.tensor([[h, w]]).float().to(device)
    # rgb_mean = torch.tensor([[0.,0.,0.]]).float().to(device)
          
    tensor = transform_image(image_bytes=image_bytes).to(device)
    outputs = model(tensor, size, rgb_mean)
    _, y_hat = outputs.max(1)
    return tensor, y_hat
