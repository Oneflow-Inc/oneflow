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


@oneflow_export("categorical_ordinal_encode")
def categorical_ordinal_encode(
    table: remote_blob_util.BlobDef,
    size: remote_blob_util.BlobDef,
    input_tensor: remote_blob_util.BlobDef,
    hash_precomputed: bool = True,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
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
    input_tensor: remote_blob_util.BlobDef,
    capacity: int,
    hash_precomputed: bool = True,
    name: str = "CategoricalOrdinalEncoder",
) -> remote_blob_util.BlobDef:
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
