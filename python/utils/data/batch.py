import random
import numpy as np
import utils.data
import itertools
import collections

class Batch(object):
    def __init__(self, dataset, num_classes=None, batch_size=64, shuffle=True, auto_reset=True, auto_fill_last=True):
        self._samples, self._labels = zip(*dataset)
        self._index = range(self.total)
        if num_classes:
            self._num_classes = num_classes
        else:
            self._num_classes = max(self._labels) + 1
        
        self._default_batch_size = batch_size
        self._shuffle = shuffle
        self._auto_reset = auto_reset
        self._auto_fill_last = auto_fill_last
        
        self.reset()
        
    def __iter__(self):
        return self
    
    @property
    def total(self):
        return len(self._labels)
    
    def samples(self, sample_index):
        return [self._samples[index] for index in sample_index]
    
    def labels(self, sample_index):
        return [self._labels[index] for index in sample_index]
    
    def has_next(self):
        return self._offset < self.total
            
    def reset(self):
        self._offset = 0
        if self._shuffle:
            random.shuffle(self._index)
        
    def notify(self):
        print 'Batch reaches the end.'
        
    def next(self, size=None):
        if self._offset >= self.total:
            if self._auto_reset:
                self.reset()
            else:
                raise StopIteration()
            
        if not size:
            size = self._default_batch_size
        if self._offset+size < self.total:
            sample_index = self._index[self._offset:self._offset+size]
            samples = self.samples(sample_index)
            labels = self.labels(sample_index)
            self._offset += size
            return samples, labels
        else:
            sample_index = self._index[self._offset:]
            samples = self.samples(sample_index)
            labels = self.labels(sample_index)
            self._offset += len(sample_index)
            while len(sample_index) < size and self._auto_fill_last:
                size -= len(sample_index)
                self.reset()
                sample_index = self._index[self._offset:self._offset+size]
                samples.extend(self.samples(sample_index))
                labels.extend(self.labels(sample_index))
                self._offset += len(sample_index)
            if not self._auto_reset:
                self._offset = self.total
                
            self.notify()
            return samples, labels
        
    def all(self):
        return self.samples(self._index), self.labels(self._index)
    
class MasterBatch(Batch):
    def __init__(self, batches, balance=None):
        num_batches = len(batches)
        if not balance:
            balance = [1./num_batches,]*num_batches # equal balance if not set
        self._batches = batches
        self._cum_balance = np.cumsum(balance)
        
    def __iter__(self):
        return self
        
    def has_next(self):
        return True
    
    def next(self, size=None, *args, **kwargs):
        rnd = random.random() * self._cum_balance[-1]
        for batch, cb in zip(self._batches, self._cum_balance):
            if cb > rnd:
                samples, labels = batch.next(size=size, *args, **kwargs)
                break
        return samples, labels
        
            
class BalancedBatch(Batch):
    def __init__(self, bases, batch_size=64, shuffle=True, auto_reset=True, auto_fill_last=True):
        self._num_classes = len(bases)
            
        #self._samples = itertools.chain(*[base._samples for base in bases])
        self._samples = [None,] * self._num_classes
        self._labels = range(self._num_classes)
        self._index = range(self._num_classes)
        
        self._bases = bases
        
        self._default_batch_size = batch_size
        self._shuffle = shuffle
        self._auto_reset = True #auto_reset # proxy must rest automatically
        self._auto_fill_last = auto_fill_last
        
        self.reset()
        
    def notify(self):
        pass
        #print 'Batch reaches the end.'
        
    def _get_from_base(self, base_list, *args, **kwargs):
        size = len(base_list)
        counter = dict(collections.Counter(base_list))
        samples_list = []
        labels_list = []
        for base_index, base_size in counter.iteritems():
            base_samples, base_labels = self._bases[base_index].next(base_size, *args, **kwargs)
            samples_list.append(base_samples)
            labels_list.append(base_labels)
        samples = np.concatenate(samples_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        return samples, labels
    
    def next(self, size=None, *args, **kwargs):
        _, base_list = Batch.next(self, size)
        return self._get_from_base(base_list, *args, **kwargs)
    
    
    def all(self, *args, **kwargs):
        _, base_list = Batch.all(self)
        return self._get_from_base(base_list, *args, **kwargs)
    
class BalancedProxyBatch(BalancedBatch):
    def __init__(self, Base, dataset, num_classes=None, batch_size=64, shuffle=True, auto_reset=True, auto_fill_last=True,
                 *args, **kwargs):
        if num_classes:
            self._num_classes = num_classes
        else:
            self._num_classes = max([e[1] for e in dataset]) + 1
            
        key = lambda e: e[1]
        dataset_sorted = sorted(dataset, key=key)
        dataset_list = [list(itr) for _, itr in itertools.groupby(dataset_sorted, key)]
        self._bases = []
        for i, sub_dataset in enumerate(dataset_list):
            base_batch = Base(sub_dataset, self._num_classes, batch_size, shuffle, auto_reset, auto_fill_last,*args, **kwargs)
            base_batch.notify = self.create_base_notify(i)
            self._bases.append(base_batch)
        
        BalancedBatch.__init__(self, self._bases, batch_size=64, shuffle=True, auto_reset=True, auto_fill_last=True)
        
    def create_base_notify(self, i):
        def base_notify():
            pass
            #print 'Class %d: Batch reaches the end.' % i
        return base_notify
    
class LoaderBatch(Batch):
    def __init__(self, dataset, num_classes=None, batch_size=64,
                 shuffle=True, auto_reset=True, auto_fill_last=True,
                 loader=None):
        Batch.__init__(self, dataset, num_classes, batch_size,
                       shuffle, auto_reset, auto_fill_last)
        self._loader = loader
        
    def _retrieve_data(self, samples, is_training):
        msg = 'Don\'t use `LoadedBatch` directly! Extend and implement `_retrieve_data` interface.'
        #raise NotImplementedError(msg)
        raise TypeError(msg)
    
    def _get_data_and_labels(self, samples, labels, is_training, one_hot, *args, **kwargs):
        cubes = self._retrieve_data(samples, is_training=is_training)
        if one_hot:
            labels = utils.data.dense_to_one_hot(labels, self._num_classes)
        return cubes, labels
        
    def next(self, size=None, is_training=False, one_hot=True, *args, **kwargs):
        samples, labels = Batch.next(self, size)
        return self._get_data_and_labels(samples, labels, is_training, one_hot, *args, **kwargs)
    
    
    def all(self, is_training=False, one_hot=True):
        samples, labels = Batch.all(self)
        return self._get_data_and_labels(samples, labels, is_training, one_hot)

class StackDataBatch(LoaderBatch):
    def load_to_data(self, loaded, is_training):
        msg = 'Don\'t use `StackDataBatch` directly! Extend and implement `load_to_data` interface.'
        #raise NotImplementedError(msg)
        raise TypeError(msg)
        
    def get_data_list(self, samples, is_training):
        data_list = []
        for sample in samples:
            loaded_sample = self._loader(sample, is_training)
            data_sample = self.load_to_data(loaded_sample, is_training)
            data_list.append(data_sample)
        return data_list
    
    def _retrieve_data(self, samples, is_training):
        data_list = self.get_data_list(samples, is_training)
        data = np.stack(data_list, axis=0)
        return data