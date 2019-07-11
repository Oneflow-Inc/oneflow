from __future__ import absolute_import

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.python.framework.inter_user_job as inter_user_job

def RemoteBlob(blob_desc.BlobDesc):
    def __init__(self, shape,
                 dtype = data_type_util.kFloat,
                 has_batch_dim = True,
                 is_dynamic = False,
                 split_axis = None,
                 broadcast = None):
        blob_desc.BlobDesc.__init__(
            self, shape, dtype, has_batch_dim, is_dynamic, split_axis, broadcast)
    
    @property
    def op_name(self):
        return self.op_name_

    def set_op_name(self, op_name):
        self.op_name_ = op_name

    def pull(self):
        return inter_user_job.pull(self)
