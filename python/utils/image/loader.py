import cv2
import utils.data.cache
import utils.image.preprocessing

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
        
class CachedImageLoader(ImageLoader):
    def __init__(self, preprocessing=None, cache=utils.data.cache.Cache()):
        if preprocessing:
            pre_stack = preprocessing._pre_stack
            training_stack = preprocessing._training_stack
            model_stack = preprocessing._model_stack
            preprocessing_b = utils.image.preprocessing.Preprocessing(pre_stack=pre_stack)
            preprocessing_c = utils.image.preprocessing.Preprocessing(training_stack=training_stack, model_stack=model_stack)
        else:
            preprocessing_b = None
            preprocessing_c = None
        ImageLoader.__init__(self, preprocessing_b) # load origin image from base class
        self._preprocessing_c = preprocessing_c
        self._cache = cache
        #print preprocessing._pre_stack, preprocessing._training_stack, preprocessing._model_stack
        #print preprocessing_b._pre_stack, preprocessing_b._training_stack, preprocessing_b._model_stack
        #print preprocessing_c._pre_stack, preprocessing_c._training_stack, preprocessing_c._model_stack
        
    def __call__(self, path, is_training=False, forced_reload=False):
        try:
            if forced_reload:
                raise KeyError()
            image = self._cache[path]
        except: # for any error, fallback to normal loader
            image = ImageLoader.__call__(self, path, is_training)
            try:
                self._cache[path] = image
            except:
                print 'Error when caching %s. Skipped.' % path
                
        if self._preprocessing_c:
            image = self._preprocessing_c(image, is_training)
        return image
        
    def reset(self):
        self._cache.clear()
        
'''
class CachedImageLoader(ImageLoader):
    def __init__(self, preprocessing=None, cache=utils.data.cache.Cache()):
        ImageLoader.__init__(self, preprocessing)
        self._cache = cache
        
    def __call__(self, path, is_training=False, forced_reload=False):
        if path not in self._cache or forced_reload:
            image = ImageLoader.__call__(self, path, is_training)
            self._cache[path] = image
        else:
            image = self._cache[path]
        return image
        
    def reset(self):
        self._cache.clear()
        
'''
