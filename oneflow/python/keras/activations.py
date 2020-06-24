from __future__ import absolute_import

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("keras.activations.relu")
def relu(x, alpha=0.0, max_value=None, threshold=0.0, name=None):
    assert alpha == 0.0
    assert max_value == None
    assert threshold == 0.0
    return flow.math.relu(x, name)


@oneflow_export("keras.activations.gelu_grad")
def gelu_grad(x, dy):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("GeluGrad_")
    setattr(op_conf.gelu_grad_conf, "x", x.unique_name)
    setattr(op_conf.gelu_grad_conf, "dy", dy.unique_name)
    op_conf.gelu_grad_conf.dx = "dx"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "dx"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("keras.activations.tanh_grad")
def tanh_grad(y, dy):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("TanhGrad_")
    setattr(op_conf.tanh_grad_conf, "y", y.unique_name)
    setattr(op_conf.tanh_grad_conf, "dy", dy.unique_name)
    op_conf.tanh_grad_conf.dx = "dx"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "dx"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("keras.activations.sigmoid")
def sigmoid(x, name=None):
    return flow.math.sigmoid(x, name)
