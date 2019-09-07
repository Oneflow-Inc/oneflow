from __future__ import absolute_import

import oneflow.core.record.record_pb2 as record_pb
import oneflow.python.lib.core.pb_util as pb_util
import numpy as np
import tempfile
import struct

def GetFirstFeatureAsNdarrayFromFile(file_path):
    for ndarray in GenearateFeatureAsNdarrayFromFile(file_path): return ndarray

def GetFirstFeatureFromFile(file_path):
    for feature in pb_util.GeneratePbMessageFromFile(file_path, record_pb.Feature): return feature

def GenearateFeatureAsNdarrayFromFile(file_path):
    for feature in pb_util.GeneratePbMessageFromFile(file_path, record_pb.Feature):
        yield Feature2Ndarray(feature)
        
def Feature2Ndarray(feature):
    if feature.HasField('float_list'):
        return np.array(list(feature.float_list.value), np.float32)
    elif feature.HasField('double_list'):
        return np.array(list(feature.double_list.value), np.float64)
    elif feature.HasField('int32_list'):
        return np.array(list(feature.int32_list.value), np.int32)
    else:
        assert False, "UNIMPLEMENTED"

def Ndarray2OFFeatureConf(ndarray):
    conf = {}
    if ndarray.dtype == np.int64: ndarray = ndarray.astype(np.int32)
    conf[_Ndarray2FeatureFieldName(ndarray)] = {'value': list(ndarray.flatten())}
    return conf

def Ndarray2OFFeatureFile(ndarray, filepath=None):
    if filepath is None: filepath = tempfile.mktemp()
    conf = Ndarray2OFFeatureConf(ndarray)
    feature = pb_util.PythonDict2PbMessage(conf, record_pb.Feature())
    f = open(filepath, 'w')
    f.write(struct.pack("q", feature.ByteSize()))
    f.write(feature.SerializeToString())
    f.close()
    return filepath

def _Ndarray2FeatureFieldName(ndarray):
    if ndarray.dtype == np.float64: return 'double_list'
    if ndarray.dtype == np.float32: return 'float_list'
    if ndarray.dtype == np.int32: return 'int32_list'
    if ndarray.dtype == np.int16: return 'int32_list'
    if ndarray.dtype == np.int8: return 'int32_list'
    assert False, "UNIMPLEMENTED: %s" % str(ndarray.dtype)
