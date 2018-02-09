import os
import random
import itertools
import numpy as np

def split_dataset_list(dataset, proportions=(1,), balanced=[], index_lists=None):
    num_samples = len(dataset)
    
    # some summary
    print 'Total:', len(dataset)
    _, labels = zip(*dataset)
    labels = np.array(labels)
    num_classes = max(labels) + 1
    for i in range(num_classes):
        num_sample = sum(labels==i)
        print 'Class %d: %d(%.00f%%) |' % (i, num_sample, float(num_sample*100)/num_samples),
    print ''
    
    # make a index list
    if index_lists:
        assert isinstance(index_lists, list) and all([isinstance(index, list) for index in index_lists]), 'Invalid index list: %s' % str(index_lists)            
    else:
        assert sum(proportions) <= 1, 'Proportions must not summed greater than 1: sum(%s)=%d' % (str(proportions), sum(proportions))
        
        # prepare balanced sets
        index_list = range(num_samples)
        random.shuffle(index_list)
        key = lambda i: labels[i]
        total_index_list = sorted(index_list, key=key)
        class_lists = [list(itr) for _, itr in itertools.groupby(total_index_list, key)]
        offsets = [0,] * num_classes
        
        balanced_index_lists = []
        for prop_index in balanced:
            # each balanced set
            prop = proportions[prop_index]
            num_per_class = int(round(float(num_samples) * prop / num_classes))
            print 'num_per_class',num_per_class
            index_list = []
            for class_index in range(num_classes):
                # each class
                r = offsets[class_index] + num_per_class
                assert r <= len(class_lists[class_index]), 'Not able to make balanced set in current situation for Dataset %d on Class %d.' % (prop_index, class_index)
                index_list.extend(class_lists[class_index][offsets[class_index]:r])
                offsets[class_index] = r
            balanced_index_lists.append(index_list)
            
        # collect the rest of samples
        rest_index_list = []
        for class_index in range(num_classes):
            rest_index_list += class_lists[class_index][offsets[class_index]:]
        random.shuffle(rest_index_list)
        
        # sample the other datasets
        index_lists = []
        offset = 0
        balanced_iter = balanced_index_lists.__iter__()
        for prop_index, prop in enumerate(proportions):
            if prop_index in balanced:
                index_lists.append(balanced_iter.next())
            else:
                r = offset + int(round(float(num_samples) * prop))
                index_lists.append(rest_index_list[offset:r])
                offset = r
    
    # some summary
    #index_list_summary(index_lists, labels)
                
    # make each dataset
    datasets = []
    for index in index_lists:
        datasets.append([dataset[i] for i in index])
    return index_lists, datasets

def index_list_summary(index_lists, labels):
    num_samples = sum([len(index_list) for index_list in index_lists])
    num_classes = max(labels) + 1
    for i, index_list in enumerate(index_lists):
        num_sample = len(index_list)
        print 'Dataset [%d]: %d(%.02f%%) :' % (i, num_sample, float(num_sample*100)/num_samples)
        dataset_label = labels[index_list]
        for ic in range(num_classes):
            num_sample = sum(dataset_label==ic)
            print 'Class [%d]: %d(%.02f%%) |' % (ic, num_sample, float(num_sample*100)/num_samples),
        print ''
        
def bar(pers, width, algin_left=False):
    fill_width = pers/100*width
    if round(fill_width) > int(fill_width):
        bar = '>' if algin_left else '<'
    else:
        bar = ''
    if algin_left:
        bar = '|' * int(fill_width) + bar
        bar = bar + (' '*(width-len(bar)))
    else:
        bar = bar + '|' * int(fill_width)
        bar = (' '*(width-len(bar)))+bar
    return bar
        
def classes_summary(dataset, num_classes, class_names=None, num_samples=None, width=20):
    num_sample = len(dataset)
    if not num_samples:
        num_samples = num_sample
    if not class_names:
        class_names = ['Class [%d]' % j for j in range(num_classes)]
    max_name_len = max([len(name) for name in class_names])
    
    _, labels = zip(*dataset)
    pers = float(num_sample*100)/num_samples
    template = '%' + str(max_name_len) + 's: %s %d (%.02f%%)'
    print template % ('Total', bar(pers, width), num_sample, pers)
    for j in range(num_classes):
        num_sample = sum(np.array(labels)==j)
        pers = float(num_sample*100)/num_samples
        print template % (class_names[j], bar(pers, width), num_sample, pers)
    
def datasets_summary(datasets, dataset_names=None, class_names=None):
    if not dataset_names:
        dataset_names = ['Dataset [%d]'%i for i in range(len(datasets))]
    alldata = list(itertools.chain(*datasets))
    _, labels = zip(*alldata)
    num_classes = max(labels) + 1
    num_samples = len(labels)
    print 'Total: _%d_ classes, _%d_ samples.'%(num_classes, num_samples)
    classes_summary(alldata, num_classes, class_names, num_samples)
    for i, dataset in enumerate(datasets):
        num_sample = len(dataset)
        print '%s: %d(%.02f%%) :' % (dataset_names[i], num_sample, float(num_sample*100)/num_samples)
        classes_summary(dataset, num_classes, class_names, num_samples)
        