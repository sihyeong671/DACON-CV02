import io
import numpy as np
from PIL import Image

import albumentations
import albumentations.pytorch
import torch

def transform_image(image_bytes: bytes) -> torch.Tensor:
    transform = albumentations.Compose([
            albumentations.Resize(height=380, width=380),
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image_array = np.array(image)
    return transform(image=image_array)['image'].unsqueeze(0)