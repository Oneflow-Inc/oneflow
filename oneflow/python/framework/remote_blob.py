from __future__ import absolute_import

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.python.framework.inter_user_job_util as inter_user_job_util
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow

class RemoteBlob(blob_desc.BlobDesc):
    def __init__(self, lbi,
                 shape = None,
                 dtype = data_type_util.kFloat,
                 has_batch_dim = True,
                 is_dynamic = False,
                 split_axis = None,
                 broadcast = None):
        blob_desc.BlobDesc.__init__(
            self, shape, dtype, has_batch_dim, is_dynamic, split_axis, broadcast)
        self.lbi_ = lbi
    
    @property
    def op_name(self):
        return self.lbi_.op_name

    @property
    def logical_blob_name(self):
        return "%s/%s" % (self.lbi_.op_name, self.lbi_.blob_name)

    def pull(self):
        return inter_user_job_util.pull(self)

    def __add__(self, rhs): 
        # TODO: scalar_add/broadcast_add/elemwise_add
        return oneflow.keras.maths.add(self, rhs)
