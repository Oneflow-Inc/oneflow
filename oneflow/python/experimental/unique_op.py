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

from typing import Optional, Tuple

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.input_blob_def as input_blob_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow_api


@oneflow_export("experimental.unique_with_counts")
def unique_with_counts(
    x: input_blob_util.ArgBlobDef,
    out_idx: flow.dtype = flow.int32,
    name: Optional[str] = None,
) -> Tuple[oneflow_api.BlobDesc]:
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("UniqueWithCounts_")
        )
        .Op("unique_with_counts")
        .Input("x", [x])
        .Attr("out_idx", out_idx)
        .Output("y")
        .Output("idx")
        .Output("count")
        .Output("num_unique")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()
