import numpy as np
from utils.data.batch import StackDataBatch
from utils.data.loader.luna16_v2 import Luna16Loader

class Luna16Batch(StackDataBatch):
    def __init__(self, dataset, num_classes=None, batch_size=64,
                 shuffle=True, auto_reset=True, auto_fill_last=True,
                 loader=Luna16Loader()):
        StackDataBatch.__init__(self, dataset, num_classes, batch_size,
                       shuffle, auto_reset, auto_fill_last,
                       loader)
        
    def load_to_data(self, loaded, is_training):
        image, image_array, cand_center, anno_center, anno_r = loaded
        return image_array
    
import threading
class RetrieveThread(threading.Thread):
    def __init__(self, loader, load_to_data, sample, is_training, data_list, index):
        threading.Thread.__init__(self)
        self._loader = loader
        self._load_to_data = load_to_data
        self._sample = sample
        self._is_training = is_training
        self._data_list = data_list
        self._index = index

    def run(self):
        loaded_sample = self._loader(self._sample, self._is_training)
        data_sample = self._load_to_data(loaded_sample, self._is_training)
        self._data_list[self._index] = data_sample

class MTBatch(StackDataBatch):
    def get_data_list(self, samples, is_training):
        data_list = [None,]*len(samples)
        
        threads = []
        for index, sample in enumerate(samples):
            t = RetrieveThread(self._loader, self.load_to_data, sample, is_training, data_list, index)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
            
        return data_list
    
import multiprocessing
def retrieve_queue(loader, load_to_data, sample, is_training, queue):
    loaded_sample = loader(sample, is_training)
    data_sample = load_to_data(loaded_sample, is_training)
    queue.put(data_sample)
    
class MPBatch(StackDataBatch):
    def __init__(self, *args, **kwargs):
        StackDataBatch.__init__(self, *args, **kwargs)
        self._queue = multiprocessing.Queue()
    
    def __del__(self):
        self._queue.close()
    
    def get_data_list(self, samples, is_training):
        procs = []
        for sample in samples:
            proc = multiprocessing.Process(target=retrieve_queue, args=(self._loader, self.load_to_data, sample, is_training, self._queue))
            proc.start()
            procs.append(proc)
            
        data_list = []
        for _ in procs:
            data_list.append(self._queue.get())
        for proc in procs:
            proc.join()
            
        return data_list
    
class MTLuna16Batch(MTBatch, Luna16Batch):
    def __init__(self, *args, **kwargs):
        MTBatch.__init__(self, *args, **kwargs)
        Luna16Batch.__init__(self, *args, **kwargs)
    
class MPLuna16Batch(MPBatch, Luna16Batch):
    def __init__(self, *args, **kwargs):
        MPBatch.__init__(self, *args, **kwargs)
        Luna16Batch.__init__(self, *args, **kwargs)
        