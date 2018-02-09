import os
import utils.datasets

def get_photo_dataset(data_path, label_path, proportions=(0.85,0.15)):
    dataset = []
    with open(label_path, 'r') as fin:
        header = fin.readline()
        assert(header == 'image,level\n')
        for line in fin:
            fname, label = line.split(',')[:2]
            dataset.append((os.path.join(data_path, fname+'.jpeg'), int(label)))
    
    _, datasets = utils.datasets.split_dataset_list(dataset, proportions=proportions)
    return datasets