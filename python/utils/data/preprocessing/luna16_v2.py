import numpy as np
import skimage.transform

def dummy_process(data):
    image, image_array, cand_center, anno_center, anno_r = data
    return image, image_array, cand_center, anno_center, anno_r

def crop_meta(data, margin=(0,0,0)):
    image, image_array, cand_center, anno_center, anno_r = data
    #print 'crop_meta(in):', cand_center, anno_center, anno_r
        
    # image_array - xyz in v2
    world_spacing = np.array(image.GetSpacing(), copy=True) # xyz
    
    r = np.array(margin, copy=True) + anno_r
    
    voxel_center = np.array(image.TransformPhysicalPointToContinuousIndex(point=cand_center), copy=True)
    voxel_r = r / world_spacing
    voxel_center = voxel_center.round()
    voxel_r = voxel_r.round()
    voxel_start = (voxel_center - voxel_r).round().astype(np.int32)
    voxel_end = (voxel_center + voxel_r).round().astype(np.int32)
    
    voxel_start[voxel_start<0] = 0
    shape = np.array(image_array.shape, copy=True)
    voxel_end[voxel_end>shape] = shape[voxel_end>shape]
    
    image_array = image_array[voxel_start[0]:voxel_end[0],
                              voxel_start[1]:voxel_end[1],
                              voxel_start[2]:voxel_end[2]]
    
    #print 'r:', r
    start = np.array(image.TransformIndexToPhysicalPoint((0,0,0)), copy=True)
    #print 'start:', start
    end = np.array(image.TransformIndexToPhysicalPoint(image_array.shape), copy=True)
    #print 'end:', end
    #print 'start + r:', start + r
    p = (start + r) - cand_center
    #print 'p:', p
    #print 'cand_center + p:', cand_center + p
    cand_center += p
    #print 'cand_center(after+=):', cand_center
    #print 'p(after+=):', p
    anno_center += p
    #print 'anno_center(after+=):', anno_center
    #print 'p(after+=):', p
        
    #print 'crop_meta(out):', cand_center, anno_center, anno_r
    return image, image_array, cand_center, anno_center, anno_r

def crop(data):
    image, image_array, cand_center, anno_center, anno_r = data
    
    #print 'crop(in):', cand_center, anno_center, anno_r
    
    # image_array - xyz in v2
    world_spacing = np.array(image.GetSpacing(), copy=True) # xyz

    voxel_center = np.array(image.TransformPhysicalPointToContinuousIndex(point=cand_center), copy=True)
    #print 'voxel_center', voxel_center
    voxel_r = anno_r / world_spacing
    #print 'voxel_r', voxel_r
    voxel_center = voxel_center.round()
    voxel_r = voxel_r.round()
    voxel_start = (voxel_center - voxel_r).round().astype(np.int32)
    #print 'voxel_start', voxel_start
    voxel_end = (voxel_center + voxel_r).round().astype(np.int32)
    #print 'voxel_end', voxel_end

    voxel_start[voxel_start<0] = 0
    shape = np.array(image_array.shape, copy=True)
    voxel_end[voxel_end>shape] = shape[voxel_end>shape]
    
    #print 'image_array.shape:', image_array.shape
    #print 'voxel_start:', voxel_start
    #print 'voxel_end:', voxel_end
    
    image_array = image_array[voxel_start[0]:voxel_end[0],
                              voxel_start[1]:voxel_end[1],
                              voxel_start[2]:voxel_end[2]]
    
    return image, image_array, cand_center, anno_center, anno_r

def resize(data, size=65):
    image, image_array, cand_center, anno_center, anno_r = data
    image_array = skimage.transform.resize(image_array, output_shape=np.array((size,size,size)),
                                    mode='symmetric', preserve_range=True,
                                    #anti_aliasing=True # supported only after 0.14.x (current dev version)
                                   )
    return image, image_array, cand_center, anno_center, anno_r

def input_normalize(data, mu=-500, sigma=300): # small sigma, sharper the curve
    image, image_array, cand_center, anno_center, anno_r = data
    image_array = (image_array - mu) / sigma
    image_array = (np.exp(image_array)-np.exp(-image_array)) / (np.exp(image_array)+np.exp(-image_array))
    return image, image_array, cand_center, anno_center, anno_r

def expand(data, r=(0,0,0)):
    image, image_array, cand_center, anno_center, anno_r = data
    anno_r = np.array(r, copy=True) + anno_r # make sure it be numpy
    return image, image_array, cand_center, anno_center, anno_r

def random_expand(data, r=[[0,0],[0,0],[0,0]]):
    r = np.array(r, copy=True)
    r = np.random.rand(3) * (r[:,1]-r[:,0]) + r[:,0]
    return expand(data, r)

def shift(data, p=(0,0,0)):
    image, image_array, cand_center, anno_center, anno_r = data
    cand_center = np.array(p, copy=True) + cand_center
    return image, image_array, cand_center, anno_center, anno_r

def random_shift(data, p=[[0,0],[0,0],[0,0]]):
    p = np.array(p, copy=True)
    p = np.random.rand(3) * (p[:,1]-p[:,0]) + p[:,0]
    return shift(data, p)
    