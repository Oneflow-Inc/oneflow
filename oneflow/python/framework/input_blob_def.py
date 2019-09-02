from __future__ import absolute_import

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as lbi_util
import oneflow.python.framework.id_util as id_util

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('input_blob_def')
class input_blob_def(blob_desc.BlobDesc):
    def __init__(self, shape,
                 dtype = data_type_util.kFloat,
                 has_batch_dim = True,
                 is_dynamic = False,
                 split_axis = None,
                 broadcast = None):
        if split_axis == None and broadcast == None: split_axis = 0
        assert type(shape) is tuple
        for dim in shape: assert type(dim) is int
        self.shape_ = shape
        self.dtype_ = dtype
        self.has_batch_dim_ = has_batch_dim
        self.is_dynamic_ = is_dynamic
        self.split_axis_ = split_axis
        self.broadcast_ = broadcast
        self.lbi_ = lbi_util.LogicalBlobId()
        self.lbi_.op_name = id_util.UniqueStr("Input_")
        self.lbi_.blob_name = "out"

    @property
    def static_shape(self): return self.shape_

    @property
    def shape(self): return self.shape_

    @property
    def dtype(self): return self.dtype_

    @property
    def has_batch_dim(self): return self.has_batch_dim_

    @property
    def split_axis(self): return self.split_axis_

    @property
    def broadcast(self): return self.broadcast_

    @property
    def is_dynamic(self): return self.is_dynamic_

    @property
    def lbi(self): return self.lbi_
        
    @property
    def op_name(self): return self.lbi_.op_name

    @property
    def blob_name(self): return self.lbi_.blob_name

    @property
    def logical_blob_name(self): return self.op_name + "/" + self.blob_name
    
    def ToInterfaceBlobConf(self):
        interface_blob_conf = op_conf_util.InterfaceBlobConf()
        interface_blob_conf.shape.dim.extend(self.shape_)
        interface_blob_conf.data_type = self.dtype_
        interface_blob_conf.has_dim0_valid_num = self.is_dynamic_
        if self.has_batch_dim_:
            interface_blob_conf.batch_axis.value = 0
        if self.is_dynamic_:
            interface_blob_conf.dim0_inner_shape.dim.extend([1,self.shape_[0]])
        assert self.split_axis_ is None or self.broadcast_ is None
        if self.split_axis_ is not None:
            interface_blob_conf.split_axis = self.split_axis_
        elif self.broadcast_ is not None:
            interface_blob_conf.broadcast = self.broadcast_
        else:
            pass # do nothing
        return interface_blob_conf
