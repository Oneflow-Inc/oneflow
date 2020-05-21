from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.hob as hob
from oneflow.python.lib.core.enable_if import enable_if

@oneflow_export("losses.add_loss", enable_if = hob.in_global_mode & hob.is_trainable)
def add_loss(loss):
    c_api_util.CurJobBuildAndInferCtx_AddLossLogicalBlobName(loss.logical_blob_name)
