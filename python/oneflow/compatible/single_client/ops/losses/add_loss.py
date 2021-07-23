from oneflow.compatible.single_client.python.framework import c_api_util as c_api_util
from oneflow.compatible.single_client.python.framework import hob as hob
from oneflow.compatible.single_client.python.eager import gradient_util as gradient_util
from oneflow.compatible.single_client.python.lib.core import enable_if as enable_if
from oneflow.compatible.single_client.python.framework import (
    remote_blob as remote_blob_util,
)
import oneflow._oneflow_internal


def api_add_loss(loss: oneflow._oneflow_internal.BlobDesc) -> None:
    """Mark a `Blob` as a loss. Auto grad starts at every loss blob. It doesn't has to be a product of typical "loss" operator like softmax loss but can also be a `Blob` produced by any operator.

    Args:
        loss: A `Blob`.
    """
    return enable_if.unique([lazy_add_loss, eager_add_loss])(loss)


@enable_if.condition(
    hob.in_global_mode & hob.is_trainable & ~hob.eager_execution_enabled
)
def lazy_add_loss(loss):
    c_api_util.CurJobBuildAndInferCtx_AddLossLogicalBlobName(loss.unique_name)


@enable_if.condition(
    hob.in_global_mode & hob.is_trainable & hob.eager_execution_enabled
)
def eager_add_loss(loss):
    c_api_util.CurJobBuildAndInferCtx_AddLossLogicalBlobName(loss.unique_name)
    gradient_util.GetDefaultBackwardBlobRegister().TrySetObject4BlobName(
        loss.logical_blob_name, loss.blob_object
    )
