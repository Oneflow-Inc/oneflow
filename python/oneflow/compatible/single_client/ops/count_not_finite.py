import os
from typing import Optional, Sequence, Union

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
    remote_blob as remote_blob_util,
)


def count_not_finite(
    x: oneflow._oneflow_internal.BlobDesc, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("CountNotFinite_")
        )
        .Op("count_not_finite")
        .Input("x", [x])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def multi_count_not_finite(
    x: Optional[Sequence[oneflow._oneflow_internal.BlobDesc]] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("MultiCountNotFinite_")
        )
        .Op("multi_count_not_finite")
        .Input("x", x)
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
