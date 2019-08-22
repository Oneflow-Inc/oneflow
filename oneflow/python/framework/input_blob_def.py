from __future__ import absolute_import

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
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
        blob_desc.BlobDesc.__init__(
            self, shape, dtype, has_batch_dim, is_dynamic, split_axis, broadcast)

    def ToInterfaceBlobConf(self):
        interface_blob_conf = op_conf_util.InterfaceBlobConf()
        interface_blob_conf.shape.dim.extend(self.shape_)
        interface_blob_conf.data_type = self.dtype_
        interface_blob_conf.has_dim0_valid_num = self.is_dynamic_
        interface_blob_conf.has_batch_dim = self.has_batch_dim_
        if self.is_dynamic_:
            interface_blob_conf.dim0_inner_shape.dim.extend([1,self.shape_[0]])
        assert self.split_axis_ is None or self.broadcast_ is None
        if self.split_axis_ is not None:
            interface_blob_conf.split_axis = self.split_axis_
        elif self.broadcast_ is not None:
            interface_blob_conf.broadcast = self.broadcast_
        else:
            pass # do nothing
        interface_blob_conf.has_batch_dim = self.has_batch_dim_
        return interface_blob_conf
