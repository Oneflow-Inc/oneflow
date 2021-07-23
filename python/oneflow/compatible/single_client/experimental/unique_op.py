from typing import Optional, Tuple

import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.core.operator import op_conf_pb2 as op_conf_util
from oneflow.compatible.single_client.core.register import (
    logical_blob_id_pb2 as logical_blob_id_util,
)
from oneflow.compatible.single_client.python.framework import (
    distribute as distribute_util,
)
from oneflow.compatible.single_client.python.framework import id_util as id_util
from oneflow.compatible.single_client.python.framework import (
    input_blob_def as input_blob_util,
)
from oneflow.compatible.single_client.python.framework import (
    interpret_util as interpret_util,
)
from oneflow.compatible.single_client.python.framework import (
    remote_blob as remote_blob_util,
)


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
