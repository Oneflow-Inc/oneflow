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
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.input_blob_def as input_blob_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow_api
from oneflow.python.oneflow_export import oneflow_export
from typing import Optional, Tuple


@oneflow_export("experimental.indexed_slices_reduce_sum")
def indexed_slices_reduce_sum(
    indices: input_blob_util.ArgBlobDef,
    values: input_blob_util.ArgBlobDef,
    name: Optional[str] = None,
) -> Tuple[oneflow_api.BlobDesc]:
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("IndexedSlicesReduceSum_")
        )
        .Op("indexed_slices_reduce_sum")
        .Input("x_indices", [indices])
        .Input("x_values", [values])
        .Output("y_indices")
        .Output("y_values")
        .Output("num_unique")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()
