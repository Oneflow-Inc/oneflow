from __future__ import absolute_import

import oneflow.python.framework.remote_blob as remote_blob_util
import numpy as np

class LocalFixedTensor(object):
    def __init__(self, ndarray = None):
        self.ndarray_ = ndarray
        
    def ndarray(self): return self.ndarray_

    def __str__(self): return str(self.ndarray_)
    
    def __getattr__(self, attr):
        return getattr(self.ndarray_, attr)

class LocalMirroredTensor(object):
    def __init__(self, ndarray_list = None):
        self.ndarray_list_ = ndarray_list
        
    def ndarray_list(self): return self.ndarray_list_
    
class LocalMirroredTensorList(object):
    def __init__(self, ndarray_lists = None):
        self.ndarray_lists_ = ndarray_lists
        
    def ndarray_lists(self): return self.ndarray_lists_

def MakeLocalBlob(ndarray_lists, consistent_blob):
    assert type(consistent_blob) is remote_blob_util.ConsistentBlob
    if consistent_blob.is_tensor_list:
        return LocalMirroredTensorList(ndarray_lists)
    assert len(ndarray_lists) == 1
    if consistent_blob.is_dynamic:
        return LocalMirroredTensor(ndarray_lists[0])
    assert len(ndarray_lists[0]) == 1
    return LocalFixedTensor(ndarray_lists[0][0])

    
def MergeLocalBlobs(local_blob_list, mirrored_blob):
    assert type(mirrored_blob) is remote_blob_util.MirroredBlob
    if mirrored_blob.is_tensor_list:
        for local_blob in local_blob_list:
            assert type(local_blob) is LocalMirroredTensorList
        return LocalMirroredTensorList([x.ndarray_lists()[0] for x in local_blob_list])
    if mirrored_blob.is_dynamic:
        for local_blob in local_blob_list:
            assert type(local_blob) is LocalMirroredTensor
        return LocalMirroredTensor([x.ndarray_list()[0] for x in local_blob_list])
    for local_blob in local_blob_list:
        assert type(local_blob) is LocalFixedTensor
        batch_axis = mirrored_blob.batch_axis
        assert type(batch_axis) is int
        ndarray = np.concatenate([x.ndarray() for x in local_blob_list], axis=batch_axis)
        return LocalFixedTensor(ndarray)

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
        return [x.ndarray() if isinstance(x, LocalFixedTensor) else x for x in args]
    return lambda self, *args: getattr(self.ndarray(), field_name)(*ConvertOtherArgs(args))

for field_name in dir(np.ndarray):
    if field_name.startswith('__') == False: continue
    if field_name in non_override_field: continue
    if hasattr(LocalFixedTensor, field_name) == False:
        setattr(LocalFixedTensor, field_name, MakeBlobMethod(field_name))
