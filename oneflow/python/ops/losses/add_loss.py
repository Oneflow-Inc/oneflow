from __future__ import absolute_import

import oneflow.python.framework.job_builder as job_builder

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("losses.add_loss")
def add_loss(loss):
    r"""Mark a `Blob` as a loss. Auto grad starts at every loss blob. It doesn't has to be a product of typical "loss" operator like softmax loss but can also be a `Blob` produced by any operator.

    Args:
        loss: A `Blob`.
    """
    job_builder.CurCtxAddLossLogicalBlobName(loss.logical_blob_name)
