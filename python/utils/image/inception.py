import utils.image
import cv2
import numpy as np

class Preprocessing(object):
    def __init__(self, width=224, height=224, central_frac=0.875):
        self._width = width
        self._height = height
        self._central_frac = central_frac
        
    def __call__(self, image):
        if image.dtype != np.float32:
            image = utils.image.image_to_float(image, dtype=np.float32)

        if self._central_frac:
            image = utils.image.central_frac_crop(image, frac=self._central_frac)

        if self._height and self._width:
            image = cv2.resize(image, (self._width, self._height), interpolation=utils.image.DEFAULT_INTER)

        image = (image - 0.5) * 2

        return image