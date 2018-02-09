from utils.data.loader.preprocessing import Preprocessing
from utils.data.cache import Cache

class CachedLoader(object):
    def __init__(self, base, base_args=[], base_kwargs={}, preprocessing=None, cache=Cache()):
        if preprocessing:
            pre_stack = preprocessing._pre_stack
            training_stack = preprocessing._training_stack
            model_stack = preprocessing._model_stack
            preprocessing_b = Preprocessing(pre_stack=pre_stack)
            preprocessing_c = Preprocessing(training_stack=training_stack, model_stack=model_stack)
        else:
            preprocessing_b = None
            preprocessing_c = None
        self.base_loader = base(preprocessing_b, *base_args, **base_kwargs) # load origin image from base class
        self._preprocessing = preprocessing_c
        self._cache = cache
        
    def get_key(self, *args, **kwargs): # arg-list should be identical to base.__call__
        return (args, kwargs)
    
    def load_to_cache(self, loaded, *args, **kwargs): # loaded is the output of base.__call__, 
        # which is to go through preprocessing stacks,
        # and could contain tuple of variable
        data = loaded
        return data
    
    def cache_to_load(self, cached, *args, **kwargs):
        # output should follow the output of base.__call__,
        # and could contain tuple of variable
        data = cached
        return data
    
    def get(self, *args, **kwargs):
        # in case `cache_to_load` is not flexible enough, override this
        key = self.get_key(*args, **kwargs)
        data = self._cache[key]
        return self.cache_to_load(data, *args, **kwargs)
    
    def put(self, data, *args, **kwargs):
        # in case `load_to_cache` is not flexible enough, override this
        key = self.get_key(*args, **kwargs)
        try:
            self._cache[key] = self.load_to_cache(data, *args, **kwargs)
        except:
            print 'Error when caching %s. Skipped.' % key
        
    def __call__(self, *args, **kwargs):
        try:
            data = self.get(*args, **kwargs)
        except: # for any error, fallback to normal loader
            data = self.load(*args, **kwargs)
        if self._preprocessing:
            data = self._preprocessing(data)
        return data
        
    def load(self, *args, **kwargs):
        # a forced reload is also possible with this function
        data = self.base_loader(*args, **kwargs)
        self.put(data, *args, **kwargs)
        return data
        
    def clear(self):
        self._cache.clear()
    