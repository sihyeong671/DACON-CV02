import albumentations as A
import numpy as np
def transform_img(image: np.array):
    transform = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.transforms.ToTensorV2()
    ])

    return transform(image=image)['image'].unsqueeze(0)