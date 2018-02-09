import os, glob
import numpy as np
import pandas as pd
import utils.datasets

def get_image_samples(image_file, cur_cands, cur_annos):
    cands_center = np.stack((cur_cands['coordX'].values,
                             cur_cands['coordY'].values,
                             cur_cands['coordZ'].values)) # (3,N)
    annos_center = np.stack((cur_annos['coordX'].values,
                             cur_annos['coordY'].values,
                             cur_annos['coordZ'].values)) # (3,M)

    cands_center_ext = np.tile(cands_center[:,np.newaxis,:], (1,annos_center.shape[1],1))
    annos_center_ext = np.tile(annos_center[:,:,np.newaxis], (1,1,cands_center.shape[1]))

    center_offset = np.linalg.norm(cands_center_ext-annos_center_ext, ord=2, axis=0)
    annos_r = np.array(cur_annos['diameter_mm'].values) / 2 # (M,)
    annos_r_ext = np.tile(annos_r[:,np.newaxis], (1,cands_center.shape[1]))

    mask = center_offset < annos_r_ext
    annos_i, cands_i = np.where(mask)

    image_samples=[]
    for cand_i, anno_i in zip(cands_i, annos_i):
        data = image_file, cands_center[:,cand_i], annos_center[:,anno_i], annos_r[anno_i]
        label = 1
        image_samples.append((data,label))
        assert cur_cands['class'].values[cand_i]==label, \
               'Label mismatch, infered %d: \n%s\n' % (label, cur_cands.iloc[[cand_i]])

    for cand_i in (set(range(cands_center.shape[1]))-set(cands_i)):
        data = image_file, cands_center[:,cand_i], None, None
        label = 0
        image_samples.append((data,label))
        assert cur_cands['class'].values[cand_i]==label, \
               'Label mismatch, infered %d: \n%s\n' % (label, cur_cands.iloc[[cand_i]])
    return image_samples

def get_luna_dataset(data_path, annos_path, cands_path, subsets):
    annos = pd.read_csv(annos_path)
    cands = pd.read_csv(cands_path)
    
    datasets = []
    for subsets_i in subsets:
        dataset = []
        for subset_i in subsets_i:
            for image_file in glob.glob(os.path.join(data_path, 'subset%d'%subset_i, '*.mhd')):
                sid = os.path.basename(image_file)[:-4]
                cur_cands = cands[cands['seriesuid']==sid]
                cur_annos = annos[annos['seriesuid']==sid]
                image_samples = get_image_samples(image_file, cur_cands, cur_annos)
                dataset += image_samples
        datasets.append(dataset)
    return datasets

def get_luna_dataset_cross(data_path, annos_path, cands_path, cross_validation):
    num_subsets = len(glob.glob(os.path.join(data_path, 'subset*')))
    
    cross_validation = (cross_validation,)
    subsets_i_train = set(range(num_subsets)) - set(cross_validation)
    subsets_i_test = set(cross_validation)
    
    return get_luna_dataset(data_path, annos_path, cands_path, (subsets_i_train, subsets_i_test))


