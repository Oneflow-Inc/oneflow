import operator
from functools import reduce
from typing import Optional

import oneflow as flow
import oneflow._oneflow_internal
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.framework.distribute as distribute_util
import oneflow.framework.id_util as id_util
import oneflow.framework.input_blob_def as input_blob_util
import oneflow.framework.interpret_util as interpret_util
import oneflow.framework.remote_blob as remote_blob_util


def square_sum(
    x: oneflow._oneflow_internal.BlobDesc, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("SquareSum_")
        )
        .Op("square_sum")
        .Input("x", [x])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
