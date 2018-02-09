import numpy as np

class Preprocessing(object):
    def __init__(self, pre_stack=[], training_stack=[], model_stack=[]):
        self._pre_stack = pre_stack
        self._training_stack = training_stack
        self._model_stack = model_stack
        
    def __call__(self, data, is_training=False, *args, **kwargs):
        for proc in self._pre_stack:
            if callable(proc):
                data = proc(data, *args, **kwargs)
                
        if is_training:
            for proc in self._training_stack:
                if callable(proc):
                    data = proc(data, *args, **kwargs)
                    
        for proc in self._model_stack:
            if callable(proc):
                data = proc(data, *args, **kwargs)
                
        return data
    
def as_np(data, dtype=np.uint8):
    data = np.array(data, dtype=dtype)
    return data

def setup(func, *ext_args, **ext_kwargs):
    def proc(data, *args, **kwargs):
        args += ext_args # could be some problem or must notice thing with the list
        kwargs.update(ext_kwargs)
        return func(data, *args, **kwargs)
    return proc
