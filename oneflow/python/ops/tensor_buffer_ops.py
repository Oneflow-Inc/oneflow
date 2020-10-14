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
    r"""This operator converts the Blob's type from TensorBuffer to Tensor. 
    Some operator's output data type is `TensorBuffer`, you can use this operator to convert back
    to `Tensor`. 

    Refer to `Concept Explanation <https://docs.oneflow.org/basics_topics/concept_explanation.html#3tensorbuffer-tensorlist>`_ 
    for more about TensorBuffer. 


    Args:
        x (BlobDef): Input `Blob`.
        dtype (dtype_util.dtype): The data dtype.
        instance_shape (Sequence[int]): The shape of each TensorBuffer instance.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        BlobDef: A `Blob`.

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def tensor_buffer_to_tensor_Job(x: tp.Numpy.Placeholder(shape=(4, 16, 64, 64), dtype=flow.float32),
        ) -> tp.Numpy:
            x = flow.tensor_to_tensor_buffer(x, 
                                            instance_dims=2)
            return flow.tensor_buffer_to_tensor(x, 
                                                instance_shape=(64, 64), 
                                                dtype=flow.float)

        x = np.random.randn(4, 16, 64, 64).astype(np.float32)
        out = tensor_buffer_to_tensor_Job(x)

        # out.shape (4, 16, 64, 64)
    
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
    r"""This operator converts the Blob's type from Tensor to TensorBuffer. 

    Refer to `Concept Explanation <https://docs.oneflow.org/basics_topics/concept_explanation.html#3tensorbuffer-tensorlist>`_ 
    for more about TensorBuffer. 


    Args:
        x (BlobDef): Input `Blob`.
        instance_dims (int): The dimensions of dynamic tensor instance. 
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        BlobDef: The result Blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def tensor_buffer_to_tensor_Job(x: tp.Numpy.Placeholder(shape=(4, 16, 64, 64), dtype=flow.float32),
        ) -> tp.Numpy:
            x = flow.tensor_to_tensor_buffer(x, 
                                            instance_dims=2)
            return flow.tensor_buffer_to_tensor(x, 
                                                instance_shape=(64, 64), 
                                                dtype=flow.float)

        x = np.random.randn(4, 16, 64, 64).astype(np.float32)
        out = tensor_buffer_to_tensor_Job(x)

        # out.shape (4, 16, 64, 64)

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
