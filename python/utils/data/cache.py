import random
from pymemcache.client.base import PooledClient
import json
import numpy as np
import zlib
import utils.data

class Cache(object):
    def __init__(self):
        self._store = {}
        
    def __getitem__(self, key):
        return self._store[key]
    
    def __setitem__(self, key, value):
        self._store[key] = value
    
    def __delitem__(self, key):
        if key in self._store:
            del self._store[key]
            
    def __len__(self):
        return len(self._store)
    
    def __iter__(self):
        return self._store.__iter__()
    
    def keys(self):
        return self._store.keys()
        
    def clear(self):
        return self._store.clear()
        
    def __del__(self):
        del self._store
        
class LimitSizeCache(Cache):
    def __init__(self, max_size=0):
        Cache.__init__(self)
        self._max_size = max_size
        
    def __setitem__(self, key, value):
        Cache.__setitem__(self, key, value)
        while self._max_size > 0 and len(self) > self._max_size:
            #print self.keys()
            pop_key = random.choice(self.keys())
            del self[pop_key]
            
class LimitMemCache(Cache):
    def __init__(self, max_mem=0):
        Cache.__init__(self)
        self._max_mem = max_mem
        
    def __setitem__(self, key, value):
        Cache.__setitem__(self, key, value)
        while self._max_mem > 0 and utils.data.getsize(self) > self._max_size:
            pop_key = random.choice(self.keys())
            del self[pop_key]
            
class MemcachedCache(Cache):
    def serialize(self, key, value):
        print key, type(value)
        if type(value) == str: # only str, not unicode or extend classes
            #print 1, len(value)
            return value, 1
        if isinstance(value, np.ndarray): # any ndarray
            #return zlib.compress(value.dumps()), 2
            #print 2, key, len(value.dumps())
            return value.dumps(), 2
        # other types
        #print 3, len(json.dumps(value))
        try:
            print json.dumps(value)
        except Exception as e:
            #print 4, e
            raise
        return json.dumps(value), 3
    
    def deserialize(self, key, value, flags):
        if flags == 1: # str
            return value
        if flags == 2: # ndarray
            #return np.loads(zlib.decompress(value))
            return np.loads(value)
        if flags == 3: # other
            return json.loads(value)

        raise TypeError("Unknown flags for value: %d" % flags)
    
    def __init__(self, server=('localhost', 11211), key_prefix='',
                 del_on_server=False, raise_on_key=False, raise_on_none=True):
        self._client = PooledClient(server, key_prefix=key_prefix,
                                    serializer=self.serialize, deserializer=self.deserialize)
        self._keys = set()
        self._del_on_server = del_on_server
        self._raise_on_key = raise_on_key
        self._raise_on_none = raise_on_none
        
    def __getitem__(self, key):
        if self._raise_on_key and key not in self._keys:
            raise KeyError
        value = self._client.get(key)
        if self._raise_on_none and value is None:
            raise KeyError
        return value
    
    def __setitem__(self, key, value):
        self._client.set(key, value)
        self._keys.add(key)
    
    def __delitem__(self, key):
        if self._del_on_server:
            self._client.delete(key)
        self._keys.discard(key)
            
    def __len__(self):
        return len(self._keys)
    
    def __iter__(self):
        return self._keys.__iter__()
    
    def keys(self):
        return list(self._keys)
        
    def clear(self):
        for key in self._keys:
            del self[key]

    def __del__(self):
        self.clear()
        self._client.close()
        
import bmemcached
class BMemcachedCache(MemcachedCache):
    def __init__(self, server=('/tmp/memcached.sock'), del_on_server=False, raise_on_key=False, raise_on_none=True, compress_level=0):
        self._client = bmemcached.Client(server)
        self._keys = set()
        self._del_on_server = del_on_server
        self._raise_on_key = raise_on_key
        self._raise_on_none = raise_on_none
        self._compress_level = compress_level
        
    def __setitem__(self, key, value):
        self._client.set(key, value, compress_level=self._compress_level)
        self._keys.add(key)
        