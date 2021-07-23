import os
from typing import Union, Optional, Sequence
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.framework.interpret_util as interpret_util
import oneflow.framework.id_util as id_util
import oneflow.framework.remote_blob as remote_blob_util
import oneflow.framework.module as module_util
import oneflow.ops.math_unary_elementwise_ops as math_unary_elementwise_ops
import oneflow._oneflow_internal


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
