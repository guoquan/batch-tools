import numpy as np
from utils.data.batch import LoaderBatch
from utils.data.loader.luna16 import Luna16Loader

class Luna16Batch(LoaderBatch):
    def __init__(self, dataset, num_classes=None, batch_size=64,
                 shuffle=True, auto_reset=True, auto_fill_last=True,
                 loader=Luna16Loader()):
        LoaderBatch.__init__(self, dataset, num_classes, batch_size,
                       shuffle, auto_reset, auto_fill_last, loader)
        
    def _retrieve_data(self, samples, is_training):
        cube_list = []
        for path, cand_center, anno_center, anno_r in samples:
            try:
                cube,_,_,_,_ = self._loader(path, cand_center, anno_center, anno_r, is_training)
            except:
                print 'Failed to load: path=%s, center=%s, is_training=%s' % (path, cand_center, is_training)
                raise
            cube_list.append(cube)
        cubes = np.stack(cube_list, axis=0)
        return cubes
    
import threading
class RetrieveThread(threading.Thread):
    def __init__(self, loader, sample, is_training, result_list, result_index):
        threading.Thread.__init__(self)
        self._loader = loader
        self._sample = sample
        self._is_training = is_training
        self._result_list = result_list
        self._result_index = result_index

    def run(self):
        self._result_list[self._result_index] = self._loader(self._sample, self._is_training)

class MTLuna16Batch(Luna16Batch):
    def _loader_wrap(self, sample, is_training):
        path, cand_center, anno_center, anno_r = sample
        cube,_,_,_,_ = self._loader(path, cand_center, anno_center, anno_r, is_training)
        return cube
        
    def _retrieve_data(self, samples, is_training):
        cube_list = [None,]*len(samples)
        
        threads = []
        for index, sample in enumerate(samples):
            t = RetrieveThread(self._loader_wrap, sample, is_training, cube_list, index)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
            
        cubes = np.stack(cube_list, axis=0)
        return cubes
    
import multiprocessing
def retrieve_queue(loader, sample, is_training, queue):
    result = loader(sample, is_training)
    queue.put(result)
    
class MPLuna16Batch(Luna16Batch):
    def __init__(self, dataset, num_classes=None, batch_size=64,
                 shuffle=True, auto_reset=True, auto_fill_last=True,
                 loader=Luna16Loader()):
        Luna16Batch.__init__(self, dataset, num_classes, batch_size, shuffle, auto_reset, auto_fill_last, loader)
        self._queue = multiprocessing.Queue()
    
    def __del__(self):
        self._queue.close()
        
    def _loader_wrap(self, sample, is_training):
        path, cand_center, anno_center, anno_r = sample
        cube,_,_,_,_ = self._loader(path, cand_center, anno_center, anno_r, is_training)
        return cube
    
    def _retrieve_data(self, samples, is_training):
        procs = []
        for sample in samples:
            proc = multiprocessing.Process(target=retrieve_queue, args=(self._loader_wrap, sample, is_training, self._queue))
            proc.start()
            procs.append(proc)
            
        cube_list = []
        for _ in procs:
            cube_list.append(self._queue.get())
        for proc in procs:
            proc.join()
            
        cubes = np.stack(cube_list, axis=0)
        return cubes
    