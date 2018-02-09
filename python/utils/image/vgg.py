import utils.image
import cv2
import numpy as np

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_DEFAULT_SIZE = 224

class Preprocessing(object):
    def __init__(self,
                 height=_DEFAULT_SIZE, width=_DEFAULT_SIZE, short=_DEFAULT_SIZE,
                 mean=(_R_MEAN, _G_MEAN, _B_MEAN)):
        self._height = height
        self._width = width
        self._short = short
        self._mean = mean
        
    def __call__(self, image):
        if self._short:
            image = utils.image.scale_short(image, self._short)
        image = utils.image.central_crop(image, self._height, self._width)
        image = utils.image.pad(image, self._height, self._width)
            
        if image.dtype != np.float32:
            image = np.array(image, dtype=np.float32)
        
        image -= self._mean

        return image