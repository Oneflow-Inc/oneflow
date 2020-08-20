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

import oneflow as flow
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.framework.id_util as id_util

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.remote_blob import BlobDef
from typing import Optional, Sequence


@oneflow_export("tensor_buffer_to_tensor")
def tensor_buffer_to_tensor(
    x: BlobDef,
    dtype: dtype_util.dtype,
    instance_shape: Sequence[int],
    name: Optional[str] = None,
) -> BlobDef:
    r"""Converts the Blob type to TensorBuffer.

    Args:
        x: Input `Blob`.
        dtype: The destination dtype.
        instance_shape: The shape of each TensorBuffer.
        name: Name for the operator.
    Returns:
        A `Blob`.
    """
    if name is None:
        name = id_util.UniqueStr("TensorBufferToTensor_")
    return (
        flow.user_op_builder(name)
        .Op("tensor_buffer_to_tensor")
        .Input("in", [x])
        .Output("out")
        .Attr("dtype", dtype)
        .Attr("instance_shape", instance_shape)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("tensor_to_tensor_buffer")
def tensor_to_tensor_buffer(
    x: BlobDef, instance_dims: int, name: Optional[str] = None,
) -> BlobDef:
    r"""Converts the TensorBuffer Blob to dense Tensor.

    Args:
        x: Input `Blob`.
        instance_dims: The number of dimensions to convert to TensorBuffer.
        name: Name for the operator.
    Returns:
        A `Blob`.
    """
    if name is None:
        name = id_util.UniqueStr("TensorToTensorBuffer_")
    return (
        flow.user_op_builder(name)
        .Op("tensor_to_tensor_buffer")
        .Input("in", [x])
        .Output("out")
        .Attr("instance_dims", instance_dims)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
