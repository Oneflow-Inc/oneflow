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

import collections
from typing import Optional, Union

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("math.reduce_mean")
def reduce_mean(
    input_blob: remote_blob_util.BlobDef,
    axis: Optional[Union[collections.Sized, int]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    reduce_sum = flow.math.reduce_sum(
        input_blob, axis=axis, keepdims=keepdims, name=name
    )
    if input_blob.is_dynamic:
        reduce_count = flow.math.reduced_shape_elem_cnt(
            input_blob, axis=axis, dtype=input_blob.dtype
        )
        return reduce_sum / reduce_count
    else:
        if axis is None:
            axes = []
        else:
            axes = list(axis) if isinstance(axis, collections.Sized) else [axis]
        reduce_count = 1
        if len(axes) == 0:
            for dim in input_blob.shape:
                reduce_count *= dim
        else:
            for i in axes:
                reduce_count *= input_blob.shape[i]
        return flow.math.multiply(reduce_sum, 1.0 / reduce_count)
