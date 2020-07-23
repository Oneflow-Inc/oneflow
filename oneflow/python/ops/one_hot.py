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

import os

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
from typing import Optional, Union


@oneflow_export("one_hot")
def one_hot(
    indices: remote_blob_util.BlobDef,
    depth: int,
    on_value: Union[int, float] = 1,
    off_value: Union[int, float] = 0,
    axis: int = -1,
    dtype: Optional[dtype_util.dtype] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    out_ndims = len(indices.shape) + 1
    if axis < 0:
        axis += out_ndims
    assert axis >= 0 and axis < out_ndims, ValueError(
        "Expected axis to between [%d, %d).  But received: %d "
        % (-out_ndims, out_ndims, axis)
    )
    out = (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("OneHot_"))
        .Op("one_hot")
        .Input("indices", [indices])
        .Attr("depth", int(depth))
        .Attr("floating_on_value", float(on_value))
        .Attr("integer_on_value", int(on_value))
        .Attr("floating_off_value", float(off_value))
        .Attr("integer_off_value", int(off_value))
        .Attr("dtype", dtype)
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
    if axis != (out_ndims - 1):
        dim_list = list(range(0, out_ndims))
        dim_list.insert(axis, out_ndims - 1)
        dim_list.pop()
        return flow.transpose(out, dim_list)
    else:
        return out
