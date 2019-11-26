from __future__ import absolute_import

import numpy as np

class LocalBlob(object):
    def __init__(self, ndarray = None):
        self.ndarray_ = ndarray
        
    def ndarray(self): return self.ndarray_

    def __getattr__(self, attr):
        return getattr(self.ndarray_, attr)

class LocalConsistentBlob(LocalBlob):
    def __init__(self, ndarray = None):
        LocalBlob.__init__(self, ndarray)
        self.lod_tree_ = None
        self.lod_ndarray_nested_list_ = None
    
    def lod_tree(self): return self.lod_tree_

    def lod_ndarray_nested_list(self): return self.lod_ndarray_nested_list_

    def set_ndarray(self, ndarray): self.ndarray_ = ndarray

    def set_lod_tree(self, lod_tree): self.lod_tree_ = lod_tree

    def set_lod_ndarray_nested_list(self, lod_ndarray_nested_list):
        self.lod_ndarray_nested_list_ = lod_ndarray_nested_list

class LocalMirrorBlob(LocalBlob):
    def __init__(self, ndarray_list):
        LocalBlob.__init__(self)
        self.ndarray_list_ = []
        self.set_ndarray_list(ndarray_list)
    
    def ndarray_list(self):
        assert len(self.ndarray_list_) > 0
        return self.ndarray_list_

    def __str__(self): return str(self.ndarray_list_)
    
    def set_ndarray_list(self, ndarray_list):
        assert len(ndarray_list) > 0
        assert len(self.ndarray_list_) == 0
        self.ndarray_list_ = ndarray_list
        self.ndarray_ = np.concatenate(self.ndarray_list_)

non_override_field = set([
    '__class__',
    '__doc__',
    '__new__',
    '__init__',
    '__del__',
    '__call__',
    '__getattr__',
    '__getattribute__',
    '__setattr__',
    '__delattr__',
    '__dir__',
    '__get__',
    '__set__',
    '__delete__',
])

def MakeBlobMethod(field_name):
    def ConvertOtherArgs(args):
        return [x.ndarray() if isinstance(x, LocalBlob) else x for x in args]
    return lambda self, *args: getattr(self.ndarray(), field_name)(*ConvertOtherArgs(args))

for field_name in dir(np.ndarray):
    if field_name.startswith('__') == False: continue
    if field_name in non_override_field: continue
    if hasattr(LocalConsistentBlob, field_name):
        setattr(LocalConsistentBlob, field_name, MakeBlobMethod(field_name))
    if hasattr(LocalMirrorBlob, field_name) == False:
        setattr(LocalMirrorBlob, field_name, MakeBlobMethod(field_name))
