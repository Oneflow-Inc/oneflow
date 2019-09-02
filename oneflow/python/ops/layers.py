from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("layers.dense")
def dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=None,
    trainable=True,
    name=None,
):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name",
            name if name is not None else id_util.UniqueStr("Dense_"))
    setattr(op_conf.fully_connected_conf, "in", inputs.logical_blob_name)
    setattr(op_conf.fully_connected_conf, "out", "out")
    setattr(op_conf.fully_connected_conf, "units", units)
    # setattr(op_conf.fully_connected_conf, "activation", activation)
    assert(activation is None)
    setattr(op_conf.fully_connected_conf, "use_bias", use_bias)
    if kernel_initializer is not None:
        op_conf.fully_connected_conf.weight_initializer.CopyFrom(
            kernel_initializer)
    if bias_initializer is not None:
        op_conf.fully_connected_conf.bias_initializer.CopyFrom(
            bias_initializer)
    setattr(op_conf, "trainable", trainable)
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)
