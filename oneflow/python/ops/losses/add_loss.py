from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if

@enable_if.condition(hob.in_global_mode & hob.is_trainable)
def add_loss(loss):
    r"""Mark a `Blob` as a loss. Auto grad starts at every loss blob. It doesn't has to be a product of typical "loss" operator like softmax loss but can also be a `Blob` produced by any operator.

    Args:
        loss: A `Blob`.
    """
    c_api_util.CurJobBuildAndInferCtx_AddLossLogicalBlobName(loss.logical_blob_name)

@oneflow_export("losses.add_loss")
def api_add_loss(loss):
    return enable_if.unique(add_loss)(loss)
