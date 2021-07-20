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

from oneflow.compatible import single_client as flow
from oneflow.core.operator import op_conf_pb2 as op_conf_util
from oneflow.core.register import logical_blob_id_pb2 as logical_blob_id_util
from oneflow.compatible.single_client.python.framework import (
    interpret_util as interpret_util,
)
from oneflow.compatible.single_client.python.framework import (
    distribute as distribute_util,
)
from oneflow.compatible.single_client.python.framework import id_util as id_util
from oneflow.compatible.single_client.python.framework import (
    input_blob_def as input_blob_util,
)
from oneflow.compatible.single_client.python.framework import (
    remote_blob as remote_blob_util,
)
from oneflow.compatible.single_client.python.oneflow_export import oneflow_export
import oneflow._oneflow_internal


@oneflow_export("experimental.unique_with_counts")
def unique_with_counts(
    x: input_blob_util.ArgBlobDef,
    out_idx: flow.dtype = flow.int32,
    name: Optional[str] = None,
) -> Tuple[oneflow._oneflow_internal.BlobDesc]:
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
