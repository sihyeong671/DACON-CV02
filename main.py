import streamlit as st
import io
import torch
import torchvision
from PIL import Image
import numpy as np

from predict import load_model, get_prediction

model = load_model()
model.eval()

st.title('Streamlit Classification Pircture')

uploaded_file = st.file_uploader('choose image', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)) # 
    st.image(image, caption='Uploaded Image')
    
    img_np = np.asarray(image)

    st.write('classify')
    _, output = get_prediction(model, img_np)
    # label = 
    st.write(output)
