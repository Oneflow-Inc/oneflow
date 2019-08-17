from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('keras.activations.relu')
def relu(x, alpha=0., max_value=None, threshold=0.):
    assert alpha == 0.
    assert max_value == None
    assert threshold == 0.
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr('Relu_')
    setattr(op_conf.relu_conf, 'in', x.logical_blob_name)
    op_conf.relu_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export('keras.activations.gelu')
def gelu(x):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr('Gelu_')
    setattr(op_conf.gelu_conf, 'in', x.logical_blob_name)
    op_conf.gelu_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export('keras.activations.tanh')
def tanh(x):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr('Tanh_')
    setattr(op_conf.tanh_conf, 'in', x.logical_blob_name)
    op_conf.tanh_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export('keras.activations.sigmoid')
def sigmoid(x):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr('Sigmoid_')
    setattr(op_conf.sigmoid_conf, 'in', x.logical_blob_name)
    op_conf.sigmoid_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

