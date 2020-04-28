from __future__ import absolute_import

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("layers.PRelu")
def prelu(
    inputs,
    alpha_initializer,
    data_format,
    channel_shared,
    name=None,
    model_distribute=distribute_util.broadcast(),
):
  channel_pos = (
        "channels_first" if data_format.startswith("NC") else "channels_last"
  )
  if channel_shared:
    alpha_shape = [1]
  else:
    if channel_pos == "channels_first":
      alpha_shape = [inputs.shape[1]]
    elif channel_pos == "channels_last":
      alpha_shape = [inputs.shape[-1]]
    else:
      raise ValueError("invalid data_format")
    alpha = flow.get_variable(
        name + "-alpha",
        shape=alpha_shape,
        dtype=inputs.dtype,
        initializer=alpha_initializer, 
        distribute=model_distribute
    )
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.prelu_conf, "in", inputs.logical_blob_name)
    setattr(op_conf.prelu_conf, "out", "out")
    setattr(op_conf.prelu_conf, "alpha", alpha.logical_blob_name)
    setattr(op_conf.prelu_conf, "data_format", channel_pos)
    setattr(op_conf.prelu_conf, "channel_shared", channel_shared)
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi) 

