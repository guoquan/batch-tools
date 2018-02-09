import numpy as np
import SimpleITK as sitk
from utils.data.cache import Cache
from utils.data.loader.cached import CachedLoader

class Luna16Loader(object):
    def __init__(self, preprocessing=None, default_r=8):
        self._preprocessing = preprocessing
        self._default_r = default_r
        
    def __call__(self, path, cand_center, anno_center, anno_r, is_training=False):
        image = sitk.ReadImage(path)
        cube = self.get_trivial_cube(image, cand_center, anno_center, anno_r)
        
        if self._preprocessing:
            cube, _, _, _, _ = self._preprocessing((cube, image, cand_center, anno_center, anno_r), is_training)
            
        return cube, image, cand_center, anno_center, anno_r
    
    def get_trivial_cube(self, image, cand_center, anno_center, anno_r):
        if not anno_r:
            anno_r = self._default_r
        #anno_r = self._default_r
        
        image_array = sitk.GetArrayFromImage(image) # zyx
        world_spacing = np.array(image.GetSpacing()) # xyz
        
        voxel_center = np.array(image.TransformPhysicalPointToContinuousIndex(point=cand_center))
        voxel_r = anno_r / world_spacing
        voxel_center = voxel_center.round()
        voxel_r = voxel_r.round()
        voxel_start = (voxel_center - voxel_r).round().astype(np.int32)
        voxel_end = (voxel_center + voxel_r).round().astype(np.int32)
        #print np.array(tuple(reversed(image_array.shape)))
        #print voxel_start
        #print (voxel_start.astype(np.float)) / np.array(tuple(reversed(image_array.shape)))
        #print voxel_end
        #print (voxel_end.astype(np.float)) / np.array(tuple(reversed(image_array.shape)))
        
        #print np.array((voxel_start,voxel_end), dtype=np.float).mean(axis=0) / np.array(tuple(reversed(image_array.shape)))
        
        voxel_start[voxel_start<0] = 0
        rshape = tuple(reversed(image_array.shape))
        for i, gt in enumerate(voxel_end>rshape):
            if gt:
                voxel_end[i] = rshape[i]
        
        cube = image_array[voxel_start[2]:voxel_end[2], voxel_start[1]:voxel_end[1], voxel_start[0]:voxel_end[0]]
        return cube
    
class Luna16CachedLoader(CachedLoader):
    def __init__(self, preprocessing=None, default_r=8, cache=Cache()):
        CachedLoader.__init__(self, Luna16Loader, [default_r,], preprocessing=preprocessing, cache=cache)
        
    def get_key(self, path, cand_center, anno_center, anno_r, is_training):
        return path
    
    def load_to_cache(self, loaded, path, cand_center, anno_center, anno_r, is_training):
        cube, image, cand_center, anno_center, anno_r = loaded
        return image
    
    def cache_to_load(self, cached, path, cand_center, anno_center, anno_r, is_training):
        image = cached
        cube = self.base_loader.get_trivial_cube(image, cand_center, anno_center, anno_r)
        return cube, image, cand_center, anno_center, anno_r
    
    
class Luna16NLoader(object):
    def __init__(self, preprocessing=None, default_r=8):
        self._preprocessing = preprocessing
        self._default_r = default_r
        
    def __call__(self, path, cand_center, anno_center, anno_r, is_training=False):
        image = sitk.ReadImage(path)
        
        if self._preprocessing:
            image, cand_center, anno_center, anno_r = self._preprocessing((image, cand_center, anno_center, anno_r), is_training)
            
        return image, cand_center, anno_center, anno_r
    
class Luna16NCachedLoader(CachedLoader):
    def __init__(self, preprocessing=None, default_r=8, cache=Cache()):
        CachedLoader.__init__(self, Luna16NLoader, [default_r,], preprocessing=preprocessing, cache=cache)
        
    def get_key(self, path, cand_center, anno_center, anno_r, is_training):
        return path
    
    def load_to_cache(self, loaded, path, cand_center, anno_center, anno_r, is_training):
        image, cand_center, anno_center, anno_r = loaded
        return image
    
    def cache_to_load(self, cached, path, cand_center, anno_center, anno_r, is_training):
        image = cached
        return image, cand_center, anno_center, anno_r
    