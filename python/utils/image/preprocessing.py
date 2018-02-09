class Preprocessing(object):
    def __init__(self, pre_stack=[], training_stack=[], model_stack=[]):
        self._pre_stack = pre_stack
        self._training_stack = training_stack
        self._model_stack = model_stack
        
    def __call__(self, image, is_training=False):
        for proc in self._pre_stack:
            if callable(proc):
                image = proc(image)
                
        if is_training:
            for proc in self._training_stack:
                if callable(proc):
                    image = proc(image)
                    
        for proc in self._model_stack:
            if callable(proc):
                image = proc(image)
        return image
    