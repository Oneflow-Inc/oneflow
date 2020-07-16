from __future__ import absolute_import

import os
from typing import Any, Optional, Union

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("keras.activations.relu")
def relu(
    x: remote_blob_util.BlobDef,
    alpha: float = 0.0,
    max_value: Optional[Any] = None,
    threshold: float = 0.0,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    assert alpha == 0.0
    assert max_value == None
    assert threshold == 0.0
    return flow.math.relu(x, name)


@oneflow_export("keras.activations.gelu_grad")
def gelu_grad(
    x: remote_blob_util.BlobDef,
    dy: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("GeluGrad_")
        )
        .Op("gelu_grad")
        .Input("x", [x])
        .Input("dy", [dy])
        .Output("dx")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("keras.activations.tanh_grad")
def tanh_grad(
    y: remote_blob_util.BlobDef,
    dy: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("TanhGrad_")
        )
        .Op("tanh_grad")
        .Input("y", [y])
        .Input("dy", [dy])
        .Output("dx")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("keras.activations.sigmoid")
def sigmoid(
    x: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    return flow.math.sigmoid(x, name)
