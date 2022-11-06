from albumentations import ImageOnlyTransform
import albumentations as A
import cv2


class SmartPad(ImageOnlyTransform):
    
    def __init__(self, always_apply=True, p=1.0):
        super(SmartPad, self).__init__(always_apply, p)
        self.always_apply = always_apply
        self.p = p
    
    def apply(self, img, **params):
        length = max(img.shape[0], img.shape[1]) 
        img = A.PadIfNeeded(
            min_height=length,
            min_width=length, 
            always_apply=self.always_apply,
            # border_mode=cv2.BORDER_CONSTANT,
            # value=0,
            p = self.p,
            )(image=img)['image']
        return img