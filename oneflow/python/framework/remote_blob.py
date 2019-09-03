from __future__ import absolute_import

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.python.framework.inter_user_job_util as inter_user_job_util
import oneflow.python.framework.job_builder as job_builder
import oneflow.python.framework.undefined as undefined
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow

class RemoteBlob(blob_desc.BlobDesc):
    def __init__(self, lbi):
        blob_desc.BlobDesc.__init__(self, lbi)
        self.job_name_ = job_builder.GetCurCtxJobName()

    @property
    def static_shape(self): return job_builder.GetStaticShape(self.job_name_, self.lbn_)

    @property
    def dtype(self): return job_builder.GetDataType(self.job_name_, self.lbn_)

    @property
    def has_batch_dim(self): return job_builder.GetHasBatchDim(self.job_name_, self.lbn_)

    @property
    def has_split_axis(self):
        if self.split_axis_ == undefined:
            return job_builder.GetHasSplitDimFromProducerView(self.job_name_, self.lbn_)
        elif type(self.split_axis_) is int:
            return True
        else:
            return False

    @property
    def split_axis(self):
        if self.split_axis_ == undefined:
            return job_builder.GetSplitDimFromProducerView(self.job_name_, self.lbn_)
        else:
            return split_axis_

    def pull(self):
        return inter_user_job_util.pull(self)

    def __add__(self, rhs):
        # TODO: scalar_add/broadcast_add/elemwise_add
        return oneflow.keras.maths.add(self, rhs)
