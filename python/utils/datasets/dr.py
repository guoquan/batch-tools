import os
import random
import cv2
import utils.datasets

def valid_photo_name(img):
    return img.startswith('C') and img.endswith('.jpg')

def valid_fa_name(img):
    return (img.startswith('GR') or img.startswith('GL')) and img.endswith('.jpg')

def valid_img_file(img_path):
    image = cv2.imread(img_path)
    return image.max() > 196

# the old dr dataset
def get_dr_dataset_per_img_list(dataset_path):
    train = []
    for eye in os.listdir(os.path.join(dataset_path,'train/PDR')):
        for img in os.listdir(os.path.join(dataset_path,'train/PDR',eye)):
            train.append((os.path.join(dataset_path,'train/PDR',eye,img),0))
          
    for eye in os.listdir(os.path.join(dataset_path,'train/NPDR')):
        for img in os.listdir(os.path.join(dataset_path,'train/NPDR',eye)):
            train.append((os.path.join(dataset_path,'train/NPDR',eye,img),1))
          
    test = []
    for eye in os.listdir(os.path.join(dataset_path,'test/PDR')):
        for img in os.listdir(os.path.join(dataset_path,'test/PDR',eye)):
            test.append((os.path.join(dataset_path,'test/PDR',eye,img),0))
          
    for eye in os.listdir(os.path.join(dataset_path,'test/NPDR')):
        for img in os.listdir(os.path.join(dataset_path,'test/NPDR',eye)):
            test.append((os.path.join(dataset_path,'test/NPDR',eye,img),1))
    return train, test

# ------------------------
# get photos in each eye
# ------------------------
def get_photo_list(dataset_path, label):
    eye_dir = os.listdir(dataset_path)
    sample_list=[]
    for eye in eye_dir:
        img_dir = os.listdir(os.path.join(dataset_path, eye))
        img_paths = [(os.path.join(dataset_path, eye, img), label) for img in img_dir if valid_photo_name(img)]
        sample_list.extend(img_paths)
    return sample_list

def get_photo_dataset(dataset_path, proportions=(0.85,0.15)):
    NORMAL = get_photo_list(os.path.join(dataset_path,'NORMAL'), 0)
    NPDR = get_photo_list(os.path.join(dataset_path,'NPDR'), 1)
    NPDR2PDR = get_photo_list(os.path.join(dataset_path,'NPDR2PDR'), 2)
    PDR = get_photo_list(os.path.join(dataset_path,'PDR'), 3)
    
    dataset = NORMAL + NPDR + NPDR2PDR + PDR
    _, datasets = utils.datasets.split_dataset_list(dataset, proportions=proportions)
    return datasets

# ------------------------
# get one photo in each eye
# ------------------------
def get_one_photo_list(dataset_path, label):
    eye_dir = os.listdir(dataset_path)
    sample_list=[]
    for eye in eye_dir:
        img_dir = os.listdir(os.path.join(dataset_path, eye))
        img_paths = [(os.path.join(dataset_path, eye, img), label) for img in img_dir if valid_photo_name(img)]
        if img_paths:
            sample_list.append(img_paths[0])
    return sample_list

def get_one_photo_dataset(dataset_path, proportions=(0.85,0.15)):
    NORMAL = get_one_photo_list(os.path.join(dataset_path,'NORMAL'), 0)
    NPDR = get_one_photo_list(os.path.join(dataset_path,'NPDR'), 1)
    NPDR2PDR = get_one_photo_list(os.path.join(dataset_path,'NPDR2PDR'), 2)
    PDR = get_one_photo_list(os.path.join(dataset_path,'PDR'), 3)
    
    dataset = NORMAL + NPDR + NPDR2PDR + PDR
    _, datasets = utils.datasets.split_dataset_list(dataset, proportions=proportions)
    return datasets

# ------------------------
# get tao's photo dataset
# ------------------------
def get_tao_photo_dataset(replace, replace_with, list_paths):
    datasets = []
    for list_path in list_paths:
        dataset = []
        with open(list_path, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                eye_path, label = line.strip().split()
                eye_path = eye_path.replace(replace, replace_with, 1)
                label = int(label)
                if eye_path.endswith('.jpg') or eye_path.endswith('.jpeg'):
                    dataset.append((eye_path, label))
                else:
                    img_dir = os.listdir(eye_path)
                    img_paths = [(os.path.join(eye_path, img), label) for img in img_dir if valid_photo_name(img)]
                    dataset.extend(img_paths)
        datasets.append(dataset)
    return datasets

# ------------------------
# get eyes
# ------------------------
def get_eye_list(dataset_path, label):
    sample_list=[]
    eye_dir = os.listdir(dataset_path)
    for eye in eye_dir:
        img_dir = os.listdir(os.path.join(dataset_path, eye))
        img_paths = [os.path.join(dataset_path, eye, img) for img in img_dir if valid_fa_name(img)]
        sample = [img_path for img_path in img_paths if valid_img_file(img_path)]
        if sample:
            sample_list.append((sample, label))
        else:
            print 'EMPTY:', os.path.join(dataset_path, eye)
    return sample_list

# the new dr dataset
def get_dr_dataset_per_eye_list(dataset_path):
    NORMAL = get_eye_list(os.path.join(dataset_path,'NORMAL'), 0)
    NPDR = get_eye_list(os.path.join(dataset_path,'NPDR'), 1)
    NPDR2PDR = get_eye_list(os.path.join(dataset_path,'NPDR2PDR'), 2)
    PDR = get_eye_list(os.path.join(dataset_path,'PDR'), 3)
    
    all_list = NORMAL + NPDR + NPDR2PDR + PDR
    random.shuffle(all_list)
    n = len(all_list)
    print 'total %d: (%d + %d + %d + %d)' % (n, len(NORMAL), len(NPDR), len(NPDR2PDR), len(PDR))
    test_part = 0.15
    part = int(round(n * test_part))
    train = all_list[part:]
    test = all_list[:part]
    return train, test