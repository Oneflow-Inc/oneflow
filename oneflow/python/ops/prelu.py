from __future__ import absolute_import

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
from oneflow.python.oneflow_export import oneflow_export
import os

@oneflow_export("layers.prelu")
def prelu(
    inputs,
    alpha_initializer=None,
    alpha_regularizer=None,
    shared_axes=None,
    trainable=True,
    name=None,
    model_distribute=distribute_util.broadcast(),
):
    alpha_shape = list(inputs.shape[1:])
    if shared_axes is not None:
      for i in shared_axes:
        assert i >= 1 and i < len(inputs.shape)
        alpha_shape[i - 1] = 1

    alpha = flow.get_variable(
        name + "-alpha",
        shape=alpha_shape,
        dtype=inputs.dtype,
        initializer=(
            alpha_initializer
            if alpha_initializer is not None
            else flow.constant_initializer(0)
        ),
        regularizer=alpha_regularizer,
        trainable=trainable,
        distribute=model_distribute
    )
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("PRelu_"))
        .Op("prelu")
        .Input("x", [inputs])
        .Input("alpha", [alpha])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
