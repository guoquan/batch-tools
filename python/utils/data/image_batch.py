import random
import numpy as np
import utils.data
import utils.data.batch
import utils.image.loader

class ImageBatch(utils.data.batch.Batch):
    def __init__(self, dataset, num_classes=None, batch_size=64, shuffle=True, auto_reset=True, auto_fill_last=True,
                 loader=utils.image.loader.ImageLoader()):
        utils.data.batch.Batch.__init__(self, dataset, num_classes, batch_size, shuffle, auto_reset, auto_fill_last)
        self._loader = loader
        
    def retrieve_images(self, path_list, is_training):
        image_list = []
        for path in path_list:
            try:
                image = self._loader(path, is_training)
            except:
                print 'Failed to load: path=%s, is_training=%s' % (path, is_training)
                raise
            image_list.append(image)
        images = np.stack(image_list, axis=0)
        return images
    
    def _get_images_and_labels(self, paths, labels, is_training, one_hot, *args, **kwargs):
        images = self.retrieve_images(paths, is_training=is_training)
        if one_hot:
            labels = utils.data.dense_to_one_hot(labels, self._num_classes)
        return images, labels
        
    def next(self, size=None, is_training=False, one_hot=True, *args, **kwargs):
        samples, labels = utils.data.batch.Batch.next(self, size)
        return self._get_images_and_labels(samples, labels, is_training, one_hot, *args, **kwargs)
    
    
    def all(self, is_training=False, one_hot=True):
        samples, labels = utils.data.batch.Batch.all(self)
        return self._get_images_and_labels(samples, labels, is_training, one_hot)
    
import multiprocessing

def retrieve_queue_func(loader, path, is_training, queue):
    image = loader(path, is_training)
    queue.put(image)
    
class MPImageBatch(ImageBatch):
    def __init__(self, dataset, num_classes=None, batch_size=64, shuffle=True, auto_reset=True, auto_fill_last=True,
                 loader=utils.image.loader.ImageLoader()):
        ImageBatch.__init__(self, dataset, num_classes, batch_size, shuffle, auto_reset, auto_fill_last, loader)
        self._queue = multiprocessing.Queue()
    
    def __del__(self):
        self._queue.close()
        
    def retrieve_images(self, path_list, is_training):
        procs = []
        for path in path_list:
            proc = multiprocessing.Process(target=retrieve_queue_func, args=(self._loader, path, is_training, self._queue))
            proc.start()
            procs.append(proc)
        image_list = []
        for _ in procs:
            image_list.append(self._queue.get())
        for proc in procs:
            proc.join()
        images = np.stack(image_list, axis=0)
        return images
'''
def retrieve_d_queue_func(qin, qout):
    loader, path, is_training = qin.get()
    image = loader(path, is_training)
    qout.put(image)
    
class DQMPImageBatch(ImageBatch):
    def __init__(self, dataset, num_classes=None, batch_size=64, shuffle=True, auto_reset=True, auto_fill_last=True,
                 loader=utils.image.loader.ImageLoader()):
        ImageBatch.__init__(self, dataset, num_classes, batch_size, shuffle, auto_reset, auto_fill_last, loader)
        self._pool = multiprocessing.Pool(processes=4)
        self._qin = multiprocessing.Queue()
        self._qout = multiprocessing.Queue()
    
    def __del__(self):
        self._qin.close()
        self._qout.close()
        self._pool.close()
        
    def retrieve_images(self, path_list, is_training):
        results = []
        for path in path_list:
            result = self._pool.apply_async(retrieve_func, (self._qout, self._qin))
        for path in path_list:
            self._qout.put((self._loader, path, is_training))
        image_list = []
        for path in path_list:
            image_list.append(self._qin.get())
        
        images = np.stack(image_list, axis=0)
        return images
    '''
'''
def retrieve_func(loader, path, is_training):
    image = loader(path, is_training)
    return image
    
class PMPImageBatch(ImageBatch):
    def __init__(self, dataset, num_classes=None, batch_size=64, shuffle=True, auto_reset=True, auto_fill_last=True,
                 loader=utils.image.loader.ImageLoader(), processes=8):
        ImageBatch.__init__(self, dataset, num_classes, batch_size, shuffle, auto_reset, auto_fill_last, loader)
        self._pool = multiprocessing.Pool(processes=processes)
        
    def __del__(self):
        self._pool.close()
        
    def retrieve_images(self, path_list, is_training):
        results = []
        for path in path_list:
            result = self._pool.apply_async(self._loader, (path, is_training))
            #result = self._pool.apply_async(retrieve_func, (self._loader, path, is_training))
            results.append(result)
        image_list = []
        for result in results:
            image_list.append(result.get())
        images = np.stack(image_list, axis=0)
        return images
'''
'''
class RetrieveProc(object):
    def __init__(self, loader):
        self._loader = loader
        
    def __call__(self, path):
        return self._loader(path, is_training)
    
class MapImageBatch(ImageBatch):
    def __init__(self, dataset, num_classes=None, batch_size=64, shuffle=True, auto_reset=True, auto_fill_last=True,
                 loader=utils.image.loader.ImageLoader(), processes=8):
        ImageBatch.__init__(self, dataset, num_classes, batch_size, shuffle, auto_reset, auto_fill_last, loader)
        self._pool = multiprocessing.Pool(processes=processes)
        self._retrieve_proc = RetrieveProc(loader)
        
    def __del__(self):
        self._pool.close()
        
    def retrieve_images(self, path_list, is_training):
        print 222
        print path_list
        image_list = self._pool.map(self._retrieve_proc, path_list)
        images = np.stack(image_list, axis=0)
        return images
    '''
import threading
class RetrieveThread(threading.Thread):
    def __init__(self, loader, path, is_training, result_list, result_index):
        threading.Thread.__init__(self)
        self._loader = loader
        self._path = path
        self._is_training = is_training
        self._result_list = result_list
        self._result_index = result_index

    def run(self):
        self._result_list[self._result_index] = self._loader(self._path, self._is_training)

class MTImageBatch(ImageBatch):
    def retrieve_images(self, path_list, is_training):
        image_list = [0,]*len(path_list)
        threads = []
        for index, path in enumerate(path_list):
            t = RetrieveThread(self._loader, path, is_training, image_list, index)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        images = np.stack(image_list, axis=0)
        return images
    
class MappedLabelImageBatch(ImageBatch):
    def __init__(self, dataset, num_classes=None, batch_size=64, shuffle=True, auto_reset=True, auto_fill_last=True,
                 loader=utils.image.loader.ImageLoader(), label_dict=None):
        ImageBatch.__init__(self, dataset, num_classes, batch_size, shuffle, auto_reset, auto_fill_last, loader)
        self._label_dict = label_dict
        
    def label_map(self, labels):
        if self._label_dict:
            labels_mapped = []
            for label in labels:
                labels_mapped.append(self._label_dict[label])
            labels_mapped = np.stack(labels_mapped, axis=0)
        else:
            return utils.data.dense_to_one_hot(labels, self._num_classes)
        return labels_mapped
    
    def _get_images_and_labels(self, paths, labels, is_training, one_hot, *args, **kwargs):
        images = self.retrieve_images(paths, is_training=is_training)
        if one_hot:
            labels = self.label_map(labels)
        return images, labels
        
class EyeBatch(ImageBatch):
    def __init__(self, dataset, num_classes=None, batch_size=4, shuffle=True, auto_reset=True, auto_fill_last=True,
                 loader=utils.image.loader.ImageLoader(), sample_size=16):
        ImageBatch.__init__(self, dataset, num_classes, batch_size, shuffle, auto_reset, auto_fill_last, loader)
        self._default_sample_size = sample_size
        
    def _get_images_and_labels(self, samples, labels, is_training, one_hot, sample_size=None):
        if not sample_size:
            sample_size = self._default_sample_size
        
        batch_images_list = []
        for sample in samples:
            n = len(sample)
            if n < sample_size:
                # too few samples, need random append
                images = self._retrieve(sample, is_training=is_training)
                while images.shape[0] < sample_size:
                    index = range(n)
                    random.shuffle(index)
                    index = index[:(sample_size-images.shape[0])]
                    rand_images = self._retrieve([sample[i] for i in index], is_training=True)
                    images = np.concatenate((images,rand_images), 0)
            else:
                # enough samples, choose randomly
                index = range(n)
                random.shuffle(index)
                index = index[:sample_size]
                images = self._retrieve([sample[i] for i in index], is_training=is_training)
            batch_images_list.append(images)
        batch_images = np.concatenate(batch_images_list, 0)
        
        if one_hot:
            labels = utils.data.dense_to_one_hot(labels, self._num_classes)
            
        return batch_images, labels