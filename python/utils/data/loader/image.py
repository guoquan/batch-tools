import cv2
from utils.data.cache import Cache
from utils.data.loader.cached import CachedLoader

class ImageLoader(object):
    def __init__(self, preprocessing=None):
        self._preprocessing = preprocessing
        
    def __call__(self, path, is_training=False):
        #print path
        image = cv2.imread(path)
        b, g, r = cv2.split(image)
        image = cv2.merge([r,g,b])
        if self._preprocessing:
            image = self._preprocessing(image, is_training)
        return image
    
class ImageCachedLoader(CachedLoader):
    def __init__(self, preprocessing=None, cache=Cache()):
        CachedLoader.__init__(self, ImageLoader, preprocessing=preprocessing, cache=cache)
        
    def get_key(self, path, is_training):
        return path