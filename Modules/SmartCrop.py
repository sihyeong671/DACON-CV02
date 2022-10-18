from albumentations import ImageOnlyTransform
import albumentations as A


class SmartCrop(ImageOnlyTransform):
    
    def __init__(self, always_apply=True, p=1.0):
        super(SmartCrop, self).__init__(always_apply, p)
        self.always_apply = always_apply
        self.p = p
    
    def apply(self, img, **params):
        length = min(img.shape[0], img.shape[1]) 
        img = A.CenterCrop(height=length, width=length, always_apply=self.always_apply, p = self.p)(image=img)['image']
        return img