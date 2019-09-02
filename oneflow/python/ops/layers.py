from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow.core.common.data_type_pb2 as data_type_conf_util

import oneflow as flow


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
    in_shape = inputs.static_shape
    in_num_axes = len(in_shape)
    assert in_num_axes >= 2
    if in_num_axes > 2:
        inputs = flow.reshape(inputs, (in_shape[0], -1))

    name_prefix = name if name is not None else id_util.UniqueStr("Dense_")
    weight = flow.get_variable(
        name="{}_weight".format(name_prefix),
        shape=(units, inputs.static_shape[1]),
        dtype=inputs.dtype,
        initializer=kernel_initializer,
        trainable=True,
        model_name="weight",
        model_split_axis=None,
    )
    out = flow.matmul(
        a=inputs, b=weight, transpose_b=True, name="{}_matmul".format(name_prefix)
    )

    if use_bias:
        # TODO: add bias_add here
        pass

    if activation is not None:
        out = activation(out)

    return out
