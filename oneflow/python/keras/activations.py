from __future__ import absolute_import

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
import os


@oneflow_export("keras.activations.relu")
def relu(x, alpha=0.0, max_value=None, threshold=0.0, name=None):
    assert alpha == 0.0
    assert max_value == None
    assert threshold == 0.0
    return flow.math.relu(x, name)


@oneflow_export("keras.activations.gelu_grad")
def gelu_grad(x, dy, name=None):
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
def tanh_grad(y, dy, name=None):
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
def sigmoid(x, name=None):
    return flow.math.sigmoid(x, name)
