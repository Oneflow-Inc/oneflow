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
