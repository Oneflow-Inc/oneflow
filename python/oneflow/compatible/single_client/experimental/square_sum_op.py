import operator
from functools import reduce
from typing import Optional

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
