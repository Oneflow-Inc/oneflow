from __future__ import absolute_import

import oneflow.python.framework.job_builder as job_builder
import oneflow.python.framework.remote_blob as remote_blob_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("losses.add_loss")
def add_loss(loss):
    if isinstance(loss, remote_blob_util.MirrorBlob):
        for sub_blob in loss.sub_consistent_blob_list:
            job_builder.CurCtxAddLossLogicalBlobName(sub_blob.logical_blob_name)
    else:
        job_builder.CurCtxAddLossLogicalBlobName(loss.logical_blob_name)
