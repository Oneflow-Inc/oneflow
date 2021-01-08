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
from typing import Optional
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow_api


@oneflow_export("categorical_ordinal_encode")
def categorical_ordinal_encode(
    table: oneflow_api.BlobDesc,
    size: oneflow_api.BlobDesc,
    input_tensor: oneflow_api.BlobDesc,
    hash_precomputed: bool = True,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator maintains a hash table to encode the categorical ordinal Blob. It converts a discrete input value into a continuous integer ID. 

    Args:
        table (oneflow_api.BlobDesc): The hash table, you can assign it as a variable. 
        size (oneflow_api.BlobDesc): The size of hash table. 
        input_tensor (oneflow_api.BlobDesc): The input Blob. 
        hash_precomputed (bool, optional): We currently only support the 'True' mode. The internal hash value will no longer be computed. Defaults to True.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np 
        import oneflow.typing as tp 

        @flow.global_function()
        def categorical_ordinal_encode_Job(x: tp.Numpy.Placeholder((3, 3), dtype=flow.int32)
        ) -> tp.Numpy: 
            dtype = x.dtype
            with flow.scope.namespace("categorical_ordinal_encode"):
                table = flow.get_variable(
                    name="Table",
                    shape=(16,),
                    dtype=dtype,
                    initializer=flow.constant_initializer(0, dtype=dtype),
                    trainable=False,
                    reuse=False,
                )
                size = flow.get_variable(
                    name="Size",
                    shape=(1,),
                    dtype=dtype,
                    initializer=flow.constant_initializer(0, dtype=dtype),
                    trainable=False,
                    reuse=False,
                )
                return flow.categorical_ordinal_encode(
                    table=table, size=size, input_tensor=x, name="Encode",
                )

        x = np.array([[7, 0, 2], 
                    [1, 7, 2], 
                    [0, 1, 7]]).astype(np.int32)

        out = categorical_ordinal_encode_Job(x)

        # out [[1 0 2]
        #      [3 1 2]
        #      [0 3 1]]

    """
    assert hash_precomputed is True
    return (
        flow.user_op_builder(name or id_util.UniqueStr("CategoricalOrdinalEncode_"))
        .Op("CategoricalOrdinalEncode")
        .Input("in", [input_tensor])
        .Input("table", [table])
        .Input("size", [size])
        .Output("out")
        .Attr("hash_precomputed", hash_precomputed)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("layers.categorical_ordinal_encoder")
def categorical_ordinal_encoder(
    input_tensor: oneflow_api.BlobDesc,
    capacity: int,
    hash_precomputed: bool = True,
    name: str = "CategoricalOrdinalEncoder",
) -> oneflow_api.BlobDesc:
    """This operator uses `oneflow.categorical_ordinal_encode` to encapsulate a categorical_ordinal_encoder. More details please refer to `oneflow.categorical_ordinal_encode`

    Args:
        input_tensor (oneflow_api.BlobDesc): The input Blob. 
        capacity (int): The capacity of hash table. 
        hash_precomputed (bool, optional): We currently only support the 'True' mode. The internal hash value will no longer be computed. Defaults to True.
        name (str, optional): The name for the operation. Defaults to "CategoricalOrdinalEncoder".

    Returns:
        oneflow_api.BlobDesc: The result Blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np 
        import oneflow.typing as tp 

        @flow.global_function()
        def categorical_ordinal_encoder_Job(x: tp.Numpy.Placeholder((3, 3), dtype=flow.int32)
        ) -> tp.Numpy: 
            return flow.layers.categorical_ordinal_encoder(x, 16)

        x = np.array([[7, 0, 2], 
                    [1, 7, 2], 
                    [0, 1, 7]]).astype(np.int32)

        out = categorical_ordinal_encoder_Job(x)

        # out [[1 0 2]
        #      [3 1 2]
        #      [0 3 1]]

    """
    assert hash_precomputed is True
    dtype = input_tensor.dtype
    with flow.scope.namespace(name):
        table = flow.get_variable(
            name="Table",
            shape=(capacity * 2,),
            dtype=dtype,
            initializer=flow.constant_initializer(0, dtype=dtype),
            trainable=False,
            reuse=False,
        )
        size = flow.get_variable(
            name="Size",
            shape=(1,),
            dtype=dtype,
            initializer=flow.constant_initializer(0, dtype=dtype),
            trainable=False,
            reuse=False,
        )
        return categorical_ordinal_encode(
            table=table, size=size, input_tensor=input_tensor, name="Encode",
        )
