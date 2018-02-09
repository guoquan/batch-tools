import cv2
import numpy as np
import utils.image
import random
import math

def scale_radius(img, scale):
    # take center row of the image (0-dim is squeezed), sum thro channels
    x = img[img.shape[0]/2,:,:].sum(1)
    # value less than 1/10*mean are background, half this len is radius of eye
    r = (x>x.mean()/10).sum() / 2
    if r < 50:
        # don't change if no radius detected
        return img
    # this scale factor `s` will make the radius of eye equals to `scale` param
    s = scale * 1.0 / r
    # scale with scale factor s: dsize = (round(s*cols), round(s*rows))
    return cv2.resize(img, (0,0), fx=s, fy=s, interpolation=utils.image.DEFAULT_INTER)

def preprocessing(img, scale=300, frac=0.9):
    #print img.shape
    # scale img to have the eye with a given radius
    a = scale_radius(img, scale)
    # subtract local mean color
    #     `GaussianBlur` gives a local mean of `scale/30` window size
    #     `addWeighted` sum the image and the blured image with wights 4 and -4, than plus 128 to each pixel
    #     In fact it gives a local constractivity of each pixel and make it 4 times by amplitude,
    #     and around background value 128.
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0,0), scale/30), -4, 128)
    # remove outer 10%
    #     prepare a mask, fill black background
    b = np.zeros(a.shape)
    #     draw a light color(1,1,1) filled circle of 90% eye area
    #     color=(1,1,1), thickness=-1(fill), lineType=8(connected line), shift=0
    cv2.circle(b, (a.shape[1]/2, a.shape[0]/2), int(scale * frac), (1, 1, 1), -1, 8, 0)
    #     outside the mask, set pixel 128; inside mask, left the pixel
    a = a * b + 128 * (1 - b)
    return a

#--------------------------
# transform image
#--------------------------
def create_rotate_mat(alpha):
    t = [[math.cos(alpha), -math.sin(alpha)],
         [math.sin(alpha), math.cos(alpha)]]
    return t

def transform_image(img, c, background_color):
    H, W, _ = img.shape
    s = [(0,0), # top-left
         (W,0), # botton-left
         (0,H), # top-right
         (W,H)] # botton-right
    s = np.array(s, dtype=np.float32).T
    
    d = np.matmul(c, s).astype(np.float32)
    
    m = d[0,:].min()
    d[0,:] -= m
    m = d[1,:].min()
    d[1,:] -= m
    W = int(math.ceil(d[0,:].max()))
    H = int(math.ceil(d[1,:].max()))
    
    c = cv2.getAffineTransform(s[:,:3].T, d[:,:3].T)
    img = cv2.warpAffine(img, c, (W, H), flags=cv2.INTER_AREA, # cv2.INTER_AREA in original code
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(background_color,background_color,background_color))
    return img

#--------------------------
# with cifar100 experiment
#--------------------------
def random_stretch_affine(c, alpha=(-0.2,0.2)):
    # ----      ------     ----
    # |  |  ->  |    |  /  |  |
    # ----      ------     |  |
    #                      ----
    
    r = min(alpha) + random.random() * (max(alpha) - min(alpha))
    t = [[1+r, 0],
         [0, 1]]
    c = np.matmul(t, c)
    r = min(alpha) + random.random() * (max(alpha) - min(alpha))
    t = [[1, 0],
         [0, 1+r]]
    c = np.matmul(t, c)
    return c

def random_flip_affine(c):
    # ----      ----
    # |* |  ->  | *|
    # ----      ----
    
    if random.random() > 0.5:
        t = [[-1, 0],
             [0, 1]]
        c = np.matmul(t, c)
    return c
        
def random_slant_affine(c, alpha=(-0.2,0.2)):
    # ----      ----       /\
    # |  |  ->  \   \  /  /  \
    # ----       ----     \  /
    #                      \/
    
    alpha = min(alpha) + random.random() * (max(alpha) - min(alpha))
    r = random.randint(3)
    if r == 1:
        t = [[1, 0],
             [alpha, 1]]
        c = np.matmul(t, c)
    elif r ==2:
        t = [[1, alpha],
             [0, 1]]
        c = np.matmul(t, c)
    elif r ==3:
        t = create_rotate_mat(alpha)
        c = np.matmul(t, c)
    return c

def distort(img, background_color=128):
    c = [[1,0],
         [0,1]]
    c = random_stretch_affine(c)
    c = random_flip_affine(c)
    c = random_slant_affine(c)
    
    img = transform_image(img, c, background_color)
    return img
    
#--------------------------
# with kaggle dr report
#--------------------------
def random_scale_affine(c, scale=(-0.1,0.1)):
    scale = min(scale) + random.random() * (max(scale) - min(scale))
    t = [[1+scale, 0],
         [0, 1+scale]]
    c = np.matmul(t, c)
    return c

def random_rotate_affine(c, alpha=(-np.pi,np.pi)):
    alpha = min(alpha) + random.random() * (max(alpha) - min(alpha))
    t = create_rotate_mat(alpha)
    c = np.matmul(t, c)
    return c

def random_skew_affine(c, alpha=(-0.2,0.2)):
    alpha = min(alpha) + random.random() * (max(alpha) - min(alpha))
    theta = (-np.pi, np.pi)
    theta = min(theta) + random.random() * (max(theta) - min(theta))
    # rotate
    t = create_rotate_mat(theta)
    c = np.matmul(t, c)
    # slant
    t = [[1, 0],
         [alpha, 1]]
    c = np.matmul(t, c)
    # rotate back
    t = create_rotate_mat(-theta)
    c = np.matmul(t, c)
    return c

def augmentation(img, background_color=128):
    c = [[1,0],
         [0,1]]
    c = random_scale_affine(c)
    c = random_rotate_affine(c)
    c = random_skew_affine(c)
    
    img = transform_image(img, c, background_color)
    return img