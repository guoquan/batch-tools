import random
import numpy as np
import cv2

#DEFAULT_INTER = cv2.INTER_LANCZOS4
DEFAULT_INTER = cv2.INTER_LINEAR


def scale(image, width=None, height=None, interpolation=DEFAULT_INTER):
    if width and height:
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    elif width:
        h, w, _ = image.shape # H=0-dim=rows=y, W=1-dim=cols=x
        r = float(width) / w
        image = cv2.resize(image, (0,0), fx=r, fy=r, interpolation=interpolation)
    elif height:
        h, w, _ = image.shape # H=0-dim=rows=y, W=1-dim=cols=x
        r = float(height) / h
        image = cv2.resize(image, (0,0), fx=r, fy=r, interpolation=interpolation)
    return image

def scale_short(image, short, interpolation=DEFAULT_INTER):
    h, w = image.shape[:2] # H=0-dim=rows=y, W=1-dim=cols=x
    if h < w:
        return scale(image, height=short, interpolation=interpolation)
    else:
        return scale(image, width=short, interpolation=interpolation)

def rotate(image, degree, interpolation=DEFAULT_INTER):
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
    image = cv2.warpAffine(image, M, (cols,rows), flags=interpolation)
    return image

INT_TYPES = [np.int8, np.int16, np.int32,
             np.uint8, np.uint16, np.uint32, np.uint64]

def image_to_float(image, dtype=np.float32):
    orig_dtype = image.dtype
    image = np.array(image, dtype=dtype)
    if orig_dtype in INT_TYPES:
        image /= np.iinfo(orig_dtype).max
    return image

def central_frac_crop(image, frac):
    rows, cols = image.shape[:2]
    off_frac = (1 - frac) / 2
    row_start = int(off_frac * rows)
    row_end = int((1 - off_frac) * rows)
    col_start = int(off_frac * cols)
    col_end = int((1 - off_frac) * cols)
    image = image[row_start:row_end, col_start:col_end, :]
    return image

def central_crop(image, width, height):
    rows, cols, _ = image.shape
    if cols > width:
        col_start = (cols - width) / 2
    else:
        col_start = 0
    col_end = cols - col_start
    if rows > height:
        row_start = (rows - height) / 2
    else:
        row_start = 0
    row_end = rows - row_start
    image = image[row_start:row_end, col_start:col_end, :]
    return image

def random_flip(image):
    if random.random() < 0.5:
        image = cv2.flip(image, 0) # flipCode=0: flipping around the x-axis
    return image

def random_crop(image, short_frac=(0.875, 1)):
    height, width = image.shape[:2]
    short = min(width, height)
    crop_size = int(short*random.uniform(*short_frac))
    row_start = int(random.uniform(0, height-crop_size))
    col_start = int(random.uniform(0, width-crop_size))
    return image[row_start:row_start+crop_size,col_start:col_start+crop_size]

def random_rotate(image, angle=(0,360)):
    image = rotate(image, random.uniform(*angle))
    return image

def remove_mask(image, thres=10000, mask_color=(0,0,0)):
    #THRES = 10000 # for 0-255 image
    #THRES = 10 # for 0-1 image
    dtype = np.array(image).dtype
    diff_image = image.astype(dtype=np.float64) # make it large enough to hold the values
    abs_diff = np.abs(diff_image - mask_color)
    
    s_cols = abs_diff.sum((0,2)) # 1-dim length vector
    s_rows = abs_diff.sum((1,2)) # 0-dim length vector
    
    image_b = image[:,s_cols>thres,:]
    image_b = image[s_rows>thres,:,:]
    if image_b.shape[0] * image_b.shape[1] < 100:
        # too small, could something be wrong
        return image
    else:
        return image_b

def remove_black_mask(image, thres=10000):
    return remove_mask(image=image, thres=thres, mask_color=(0,0,0))

def pad(image,width=None, height=None, pad_color=(0,0,0)):
    h, w, c = image.shape
    if not width or width < w:
        width = w
    if not height or height < h:
        height = h
    new_image = np.ones((width, height, c)) * pad_color
    h_offset = int((height-h)/2)
    w_offset = int((width-w)/2)
    new_image[h_offset:h_offset+h,w_offset:w_offset+w,:] = image
    new_image = new_image.astype(image.dtype)
    return new_image

def square_pad(image, pad_color=(0,0,0)):
    h, w, c = image.shape
    size = max(h,w)
    new_image = np.ones((size, size, c)) * pad_color
    new_image = new_image.astype(image.dtype)
    h_offset = int((size-h)/2)
    w_offset = int((size-w)/2)
    new_image[h_offset:h_offset+h,w_offset:w_offset+w,:] = image
    return new_image

def as_np(image, dtype=np.uint8):
    image = np.array(image, dtype=dtype)
    return image

def setup(func, **kwargs):
    return lambda img: func(img, **kwargs)