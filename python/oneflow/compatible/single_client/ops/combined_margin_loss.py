import os
from typing import Optional, Sequence, Union

import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.core.operator import op_conf_pb2 as op_conf_util
from oneflow.compatible.single_client.core.register import (
    logical_blob_id_pb2 as logical_blob_id_util,
)
from oneflow.compatible.single_client.python.framework import id_util as id_util
from oneflow.compatible.single_client.python.framework import (
    interpret_util as interpret_util,
)
from oneflow.compatible.single_client.python.framework import module as module_util
from oneflow.compatible.single_client.python.framework import (
    remote_blob as remote_blob_util,
)
from oneflow.compatible.single_client.python.ops import (
    math_unary_elementwise_ops as math_unary_elementwise_ops,
)


def combined_margin_loss(
    x: oneflow._oneflow_internal.BlobDesc,
    label: oneflow._oneflow_internal.BlobDesc,
    m1: float = 1,
    m2: float = 0,
    m3: float = 0,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    depth = x.shape[1]
    (y, theta) = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("CombinedMarginLoss_")
        )
        .Op("combined_margin_loss")
        .Input("x", [x])
        .Input("label", [label])
        .Output("y")
        .Output("theta")
        .Attr("m1", float(m1))
        .Attr("m2", float(m2))
        .Attr("m3", float(m3))
        .Attr("depth", int(depth))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return y
