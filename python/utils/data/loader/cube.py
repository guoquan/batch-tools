import numpy as np
import skimage.transform

def dummy_process(data):
    cube, image, cand_center, anno_center, anno_r = data
    return cube, image, cand_center, anno_center, anno_r

def resize(data, size=65):
    cube, image, cand_center, anno_center, anno_r = data
    #print cube.shape
    cube = skimage.transform.resize(cube, output_shape=np.array((size,size,size)),
                                    mode='symmetric', preserve_range=True,
                                    #anti_aliasing=True # supported only after 0.14.x (current dev version)
                                   )
    return cube, image, cand_center, anno_center, anno_r

def input_normalize(data, mu=-300, sigma=300): # small sigma, sharper the curve
    #print 'aaa'
    cube, image, cand_center, anno_center, anno_r = data
    #print 'bbb'
    cube = (cube - mu) / sigma
    #print 'ccc'
    #print cube
    #cube[cube>709.783] = 709.783
    cube = np.tanh(cube)
    #print 'ddd'
    return cube, image, cand_center, anno_center, anno_r
