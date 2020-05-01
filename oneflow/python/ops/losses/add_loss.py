from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.g_func_ctx as g_func_ctx


@oneflow_export("losses.add_loss")
def add_loss(loss):
    g_func_ctx.CurJobBuildAndInferCtx_AddLossLogicalBlobName(loss.logical_blob_name)
