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
    # TODO: check inputs' rank >= 2
    # TODO: reshape if inputs' rank > 2
    # TODO: check shape in inputs and units

    # TODO: set variable's dtype = inputs' dtype
    initializer = op_conf_util.InitializerConf()
    initializer.random_uniform_conf.min = 0
    initializer.random_uniform_conf.max = 10
    weight = flow.get_variable(
        name="dense_weight",
        shape=(units, 32),
        dtype=data_type_conf_util.kFloat,
        initializer=initializer,
        trainable=True,
        model_name="dense-weight",
        model_split_axis=None,
    )
    out = flow.matmul(a=inputs, b=weight, transpose_b=True)

    if use_bias:
        # TODO: set variable's dtype = inputs' dtype
        # flow.get_variable(shape=(128), dtype=dtat_type_conf_util.kFloat)
        # TODO: add bias_add here
        pass

    if activation is not None:
        out = activation(out)       

    return out
