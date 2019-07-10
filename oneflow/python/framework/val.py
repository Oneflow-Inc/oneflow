from __future__ import absolute_import

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.core.common.data_type_pb2 as data_type_util

class val(blob_desc.BlobDesc):
    def __init__(self, shape,
                 dtype = data_type_util.kFloat,
                 has_batch_dim = True,
                 is_dynamic = False,
                 split_axis = None,
                 broadcast = None):
        blob_desc.BlobDesc.__init__(
            self, shape, dtype, has_batch_dim, is_dynamic, split_axis, broadcast)
