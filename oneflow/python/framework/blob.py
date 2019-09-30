from __future__ import absolute_import

import numpy as np

class Blob(object):
    def __init__(self, ndarray = None):
        self.ndarray_ = ndarray
        self.lod_tree_ = None
        self.lod_ndarray_nested_list_ = None
    
    def ndarray(self): return self.ndarray_

    def lod_tree(self): return self.lod_tree_

    def lod_ndarray_nested_list(self): return self.lod_ndarray_nested_list_

    def set_ndarray(self, ndarray): self.ndarray_ = ndarray

    def set_lod_tree(self, lod_tree): self.lod_tree_ = lod_tree

    def set_lod_ndarray_nested_list(self, lod_ndarray_nested_list):
        self.lod_ndarray_nested_list_ = lod_ndarray_nested_list

    def __getattr__(self, attr):
        return getattr(self.ndarray_, attr)

no_override_field = set([
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
        return [x.ndarray_ if isinstance(x, Blob) else x for x in args]
    return lambda self, *args: getattr(self.ndarray_, field_name)(*ConvertOtherArgs(args))

for field_name in dir(np.ndarray):
    if field_name.startswith('__') == False: continue
    if field_name in no_override_field: continue
    setattr(Blob, field_name, MakeBlobMethod(field_name))
