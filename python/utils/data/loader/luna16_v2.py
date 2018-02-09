import numpy as np
import SimpleITK as sitk
from utils.data.cache import Cache
from utils.data.loader.cached import CachedLoader

class Luna16Loader(object):
    def __init__(self, preprocessing=None, default_r=8):
        self._preprocessing = preprocessing
        self._default_r = default_r
        
    def __call__(self, sample, is_training=False):
        path, cand_center, anno_center, anno_r = sample
        # make sure using a copy
        cand_center = np.array(cand_center, copy=True)
        anno_center = np.array(anno_center, copy=True)
        anno_r = np.array(anno_r, copy=True)
        
        if not anno_r:
            assert not anno_center
            anno_r = np.array(self._default_r, copy=True)
            anno_center = np.array(cand_center, copy=True)
            
        #import time
        #start = time.time()
        image = sitk.ReadImage(path)
        #print 'image = sitk.ReadImage(path): %f' % (time.time() - start)
        #start = time.time()
        image_array = np.array(sitk.GetArrayFromImage(image), copy=True) # zyx
        #print 'image_array = sitk.GetArrayFromImage(image): %f' % (time.time() - start)
        #start = time.time()
        image_array = image_array.transpose(2,1,0)
        #print 'image_array = image_array.transpose(2,1,0): %f' % (time.time() - start)
        sample = image, image_array, cand_center, anno_center, anno_r
        
        if self._preprocessing:
            try:
                sample = self._preprocessing(sample, is_training)
            except Exception as e:
                print 'Error when preprocessing %s @ (%s)' % (path, str(cand_center))
                print e
                raise
            
        return sample
    
class Luna16CachedLoader(CachedLoader):
    def __init__(self, preprocessing=None, default_r=8, cache=Cache()):
        CachedLoader.__init__(self, Luna16Loader, [default_r,], preprocessing=preprocessing, cache=cache)
        
    def get_key(self, sample, is_training):
        path, cand_center, anno_center, anno_r = sample
        return path
    
    def load_to_cache(self, loaded, sample, is_training):
        image, image_array, cand_center, anno_center, anno_r = loaded
        return image, image_array
    
    def cache_to_load(self, cached, sample, is_training):
        image, image_array = cached
        path, cand_center, anno_center, anno_r = sample
        if not anno_r:
            assert not anno_center
            anno_r = self._default_r
            anno_center = cand_center
            
        sample = image, image_array, cand_center, anno_center, anno_r
        return sample
    