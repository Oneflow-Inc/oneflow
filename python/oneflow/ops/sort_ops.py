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
from typing import Optional

import oneflow as flow
import oneflow._oneflow_internal
import oneflow.framework.id_util as id_util
import oneflow.framework.remote_blob as remote_blob_util
from oneflow.ops.transpose_util import (
    get_inversed_perm,
    get_perm_when_transpose_axis_to_last_dim,
)


def _sort_at_last_dim(
    input: oneflow._oneflow_internal.BlobDesc,
    direction: str = "ASCENDING",
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    assert direction in ["ASCENDING", "DESCENDING"]
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Sort_"))
        .Op("sort")
        .Input("in", [input])
        .Output("out")
        .Attr("direction", direction)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def _argsort_at_last_dim(
    input: oneflow._oneflow_internal.BlobDesc,
    direction: str = "ASCENDING",
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    assert direction in ["ASCENDING", "DESCENDING"]
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("ArgSort_")
        )
        .Op("arg_sort")
        .Input("in", [input])
        .Output("out")
        .Attr("direction", direction)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
