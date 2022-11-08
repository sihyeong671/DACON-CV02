import albumentations as A
from albumentations import ImageOnlyTransform
from albumentations.pytorch.transforms import ToTensorV2


class SmartCrop(ImageOnlyTransform):
    
    def __init__(self, always_apply=True, p=1.0):
        super(SmartCrop, self).__init__(always_apply, p)
        self.always_apply = always_apply
        self.p = p
    
    def apply(self, img, **params):
        length = min(img.shape[0], img.shape[1]) 
        img = A.CenterCrop(height=length, width=length, always_apply=self.always_apply, p = self.p)(image=img)['image']
        return img

class PreProcessing(ImageOnlyTransform):
    def __init__(self):
        super(PreProcessing, self).__init__()
        self.transform = None
    
    def apply(self, img, **params):
        if self.transform != None:
            img = self.transform(image=img)['image']
        return img

class PostProcessing(ImageOnlyTransform):
    def __init__(self, input_width:int,input_height:int):
        super(PostProcessing, self).__init__()
        self.transform = A.Compose([
                            A.Resize(input_height, input_width),
                            A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])
    
    def apply(self, img, **params):
        if self.transform != None:
            img = self.transform(image=img)['image']
        return img


    