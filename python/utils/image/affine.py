import cv2
import numpy as np
import utils.image
import random
import math

def apply(img, t, background_color=(0,0,0)):
    H, W, _ = img.shape # H=0-dim=rows=y, W=1-dim=cols=x
    s = [(0,0), # top-left
         (W,0), # top-right
         (0,H), # botton-left
         (W,H)] # botton-right
    s = np.array(s, dtype=np.float32).T
    
    d = np.matmul(t, s).astype(np.float32)
    
    d[0,:] -= d[0,:].min() 
    d[1,:] -= d[1,:].min()
    W = int(math.ceil(d[0,:].max()))
    H = int(math.ceil(d[1,:].max()))
    
    t = cv2.getAffineTransform(s[:,:3].T, d[:,:3].T)
    img = cv2.warpAffine(img, t, (W, H), flags=utils.image.DEFAULT_INTER,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=background_color)
    return img

def stack(*args):
    t = get_ident()
    for func in args:
        if callable(func):
            t = func(t)
    return t

def setup(func, **kwargs):
    return lambda img: func(img, **kwargs)

#-----------------------------
# useful affine transforms
#-----------------------------
def get_ident():
    t = [[1, 0],
         [0, 1]]
    return t

def get_rotate(alpha):
    t = [[math.cos(alpha), -math.sin(alpha)],
         [math.sin(alpha), math.cos(alpha)]]
    return t

def rotate(c, alpha=0):
    # ----       /\
    # |  |  ->  /  \
    # ----     \  /
    #           \/
    t = get_rotate(alpha)
    c = np.matmul(t, c)
    return c

def flip_h(c):
    # ----      ----
    # |* |  ->  | *|
    # |  |  ->  |  |
    # ----      ----
    t = [[-1, 0],
         [0, 1]]
    c = np.matmul(t, c)
    return c

def flip_v(c):
    # ----      ----
    # |* |  ->  |  |
    # |  |  ->  |* |
    # ----      ----
    t = [[-1, 0],
         [0, 1]]
    c = np.matmul(t, c)
    return c

def resize(c, scale=1):
    # ---      ------
    # | |  ->  |    |
    # ---      |    |
    #          ------
    t = [[scale, 0],
         [0, scale]]
    c = np.matmul(t, c)
    return c

def stretch_h(c, scale=1):
    # ----      ------
    # |  |  ->  |    |
    # ----      ------
    t = [[scale, 0],
         [0, 1]]
    c = np.matmul(t, c)
    return c

def stretch_v(c, scale=1):
    # ----      ----
    # |  |  ->  |  |
    # ----      |  |
    #           ----
    t = [[1, 0],
         [0, scale]]
    c = np.matmul(t, c)
    return c

def slant_h(c, alpha=0):
    # ----      ----
    # |  |  ->  \   \
    # ----       ----
    t = [[1, alpha],
         [0, 1]]
    c = np.matmul(t, c)
    return c

def slant_v(c, alpha=0):
    # ----       /|
    # |  |  ->  / |
    # ----     | /
    #          |/
    t = [[1, 0],
         [alpha, 1]]
    c = np.matmul(t, c)
    return c

def random_resize(c, scale=(0.9,1.1)):
    c = resize(c, random.uniform(*scale))
    return c
    
def random_stretch(c, scale=(0.9,1.1)):
    c = stretch_h(c, random.uniform(*scale))
    c = stretch_v(c, random.uniform(*scale))
    return c

def random_flip(c):
    if random.random > 0.5:
        flip = random.choice([flip_h, flip_v])
        c = flip(c)
    return c

def random_slant(c, alpha=(-0.2,0.2)):
    alpha = min(alpha) + random.random() * (max(alpha) - min(alpha))
    slant = random.choice([slant_h, slant_v])
    c = slant(c, alpha)
    return c

def random_rotate(c, alpha=(-np.pi,np.pi)):
    c = rotate(c, random.uniform(*alpha))
    return c

def random_skew(c, alpha=(-0.2,0.2)):
    theta = random.uniform(-np.pi,np.pi)
    # rotate
    c = rotate(c, theta)
    # slant
    c = random_slant(c, alpha)
    # rotate back
    c = rotate(c, -theta)
    return c
