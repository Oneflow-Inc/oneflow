"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.hob as hob
import oneflow.python.eager.gradient_util as gradient_util
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow_api


@oneflow_export("losses.add_loss")
def api_add_loss(loss: oneflow_api.BlobDesc) -> None:
    r"""Mark a `Blob` as a loss. Auto grad starts at every loss blob. It doesn't has to be a product of typical "loss" operator like softmax loss but can also be a `Blob` produced by any operator.

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
