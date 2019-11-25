from __future__ import absolute_import

import oneflow.python.framework.job_builder as job_builder
import oneflow.python.framework.remote_blob as remote_blob_util

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("losses.add_loss")
def add_loss(loss):
    job_builder.CurCtxAddLossLogicalBlobName(loss.logical_blob_name)
