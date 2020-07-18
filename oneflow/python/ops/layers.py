from __future__ import absolute_import

import os
from typing import Callable, Optional, Union, Tuple, List
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("layers.dense")
def dense(
    inputs: remote_blob_util.BlobDef,
    units: int,
    activation: Optional[remote_blob_util.BlobDef] = None,
    use_bias: bool = True,
    kernel_initializer: Optional[op_conf_util.InitializerConf] = None,
    bias_initializer: Optional[op_conf_util.InitializerConf] = None,
    kernel_regularizer: Optional[op_conf_util.RegularizerConf] = None,
    bias_regularizer: Optional[op_conf_util.RegularizerConf] = None,
    trainable: bool = True,
    name: Optional[str] = None,
    model_distribute: distribute_util.Distribute = distribute_util.broadcast(),
) -> remote_blob_util.BlobDef:
    r"""
    Analogous to `tf.keras.layers.Dense <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`_

    """
    in_shape = inputs.shape
    in_num_axes = len(in_shape)
    assert in_num_axes >= 2

    name_prefix = name if name is not None else id_util.UniqueStr("Dense_")
    inputs = flow.reshape(inputs, (-1, in_shape[-1])) if in_num_axes > 2 else inputs

    assert (
        model_distribute is distribute_util.auto()
        or model_distribute is distribute_util.broadcast()
        or model_distribute is distribute_util.split(0)
    )

    if model_distribute is distribute_util.split(0):
        assert in_num_axes == 2  # model distribute is hard for reshape split dim 1

    weight = flow.get_variable(
        name="{}-weight".format(name_prefix),
        shape=(units, inputs.shape[1]),
        dtype=inputs.dtype,
        initializer=(
            kernel_initializer
            if kernel_initializer is not None
            else flow.constant_initializer(0)
        ),
        regularizer=kernel_regularizer,
        trainable=trainable,
        model_name="weight",
        distribute=model_distribute,
    )
    weight = weight.with_distribute(model_distribute)

    out = flow.matmul(
        a=inputs, b=weight, transpose_b=True, name="{}_matmul".format(name_prefix),
    )
    if use_bias:
        bias = flow.get_variable(
            name="{}-bias".format(name_prefix),
            shape=(units,),
            dtype=inputs.dtype,
            initializer=(
                bias_initializer
                if bias_initializer is not None
                else flow.constant_initializer(0)
            ),
            regularizer=bias_regularizer,
            trainable=trainable,
            model_name="bias",
            distribute=model_distribute,
        )
        bias = bias.with_distribute(model_distribute)
        out = flow.nn.bias_add(out, bias, name="{}_bias_add".format(name_prefix))
    out = (
        activation(out, name="{}_activation".format(name_prefix))
        if activation is not None
        else out
    )
    out = flow.reshape(out, in_shape[:-1] + (units,)) if in_num_axes > 2 else out

    return out


@oneflow_export("layers.conv2d")
def conv2d(
    inputs: remote_blob_util.BlobDef,
    filters: int,
    kernel_size: Union[int, List[int], Tuple[int]] = 1,
    strides: Union[int, List[int], Tuple[int]] = 1,
    padding: str = "VALID",
    data_format: str = "NCHW",
    dilation_rate: int = 1,
    groups: int = 1,
    activation: Optional[
        Callable[[remote_blob_util.BlobDef, str], remote_blob_util.BlobDef]
    ] = None,
    use_bias: bool = True,
    kernel_initializer: Optional[op_conf_util.InitializerConf] = None,
    bias_initializer: Optional[op_conf_util.InitializerConf] = None,
    kernel_regularizer: Optional[op_conf_util.RegularizerConf] = None,
    bias_regularizer: Optional[op_conf_util.RegularizerConf] = None,
    trainable: bool = True,
    name: Optional[str] = None,
    weight_name: Optional[str] = None,
    bias_name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    name_prefix = name if name is not None else id_util.UniqueStr("Conv2D_")
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    else:
        assert isinstance(kernel_size, (list, tuple))
        kernel_size = tuple(kernel_size)

    assert isinstance(groups, int)
    assert groups > 0
    assert groups <= filters
    assert filters % groups == 0
    if data_format.upper() == "NCHW":
        assert groups <= inputs.shape[1]
        assert inputs.shape[1] % groups == 0
        weight_shape = (filters, inputs.shape[1] // groups) + kernel_size
    elif data_format.upper() == "NHWC":
        assert groups == 1
        assert groups <= inputs.shape[3]
        assert inputs.shape[3] % groups == 0
        weight_shape = (
            filters,
            kernel_size[0],
            kernel_size[1],
            inputs.shape[3] // groups,
        )
    else:
        raise ValueError("data_format must be in NCHW or NHWC")

    weight = flow.get_variable(
        weight_name if weight_name else name_prefix + "-weight",
        shape=weight_shape,
        dtype=inputs.dtype,
        initializer=kernel_initializer
        if kernel_initializer is not None
        else flow.constant_initializer(0),
        regularizer=kernel_regularizer,
        trainable=trainable,
        model_name="weight",
    )

    output = flow.nn.conv2d(
        inputs,
        weight,
        strides,
        padding,
        data_format,
        dilation_rate,
        groups=groups,
        name=name,
    )
    if use_bias:
        bias = flow.get_variable(
            bias_name if bias_name else name_prefix + "-bias",
            shape=(filters,),
            dtype=inputs.dtype,
            initializer=bias_initializer
            if bias_initializer is not None
            else flow.constant_initializer(0),
            regularizer=bias_regularizer,
            trainable=trainable,
            model_name="bias",
        )
        output = flow.nn.bias_add(
            output, bias, data_format, name=name_prefix + "-bias_add"
        )
    if activation is not None:
        output = activation(output, name=name_prefix + "-activation")

    return output


@oneflow_export("layers.layer_norm")
def layer_norm(
    inputs: remote_blob_util.BlobDef,
    center: bool = True,
    scale: bool = True,
    trainable: bool = True,
    begin_norm_axis: int = 1,
    begin_params_axis: int = -1,
    epsilon: float = 1e-5,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""
    Analogous to `tf.keras.layers.LayerNormalization <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization>`_

    """
    name = name if name is not None else id_util.UniqueStr("LayerNorm_")
    op = (
        flow.user_op_builder(name)
        .Op("layer_norm")
        .Input("x", [inputs])
        .Output("y")
        .Output("mean")
        .Output("inv_variance")
    )
    if center == False and scale == False:
        trainable = False
    param_shape = inputs.shape[begin_params_axis:]
    if center:
        beta = flow.get_variable(
            name="{}-beta".format(name),
            shape=param_shape,
            dtype=inputs.dtype,
            initializer=flow.constant_initializer(0.0),
            trainable=trainable,
            model_name="beta",
            distribute=distribute_util.broadcast(),
        )
        op.Input("beta", [beta])
    if scale:
        gamma = flow.get_variable(
            name="{}-gamma".format(name),
            shape=param_shape,
            dtype=inputs.dtype,
            initializer=flow.constant_initializer(1.0),
            trainable=trainable,
            model_name="gamma",
            distribute=distribute_util.broadcast(),
        )
        op.Input("gamma", [gamma])
        op.Output("normalized")
    op.Attr("center", center, "AttrTypeBool")
    op.Attr("scale", scale, "AttrTypeBool")
    op.Attr("begin_norm_axis", begin_norm_axis, "AttrTypeInt64")
    op.Attr("begin_params_axis", begin_params_axis, "AttrTypeInt64")
    op.Attr("epsilon", epsilon, "AttrTypeDouble")
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("layers.layer_norm_grad")
def layer_norm_grad(
    dy: remote_blob_util.BlobDef,
    x: remote_blob_util.BlobDef,
    mean: remote_blob_util.BlobDef,
    inv_variance: remote_blob_util.BlobDef,
    begin_norm_axis: int = 1,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    name = name if name is not None else id_util.UniqueStr("LayerNormGrad_")
    op = (
        flow.user_op_builder(name)
        .Op("layer_norm_grad")
        .Input("dy", [dy])
        .Input("x", [x])
        .Input("mean", [mean])
        .Input("inv_variance", [inv_variance])
        .Output("dx")
        .Attr("begin_norm_axis", begin_norm_axis, "AttrTypeInt64")
        .Attr("epsilon", 1e-5, "AttrTypeDouble")
    )
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("layers.layer_norm_param_grad")
def layer_norm_param_grad(
    dy: remote_blob_util.BlobDef,
    norm: remote_blob_util.BlobDef,
    gamma: remote_blob_util.BlobDef,
    begin_params_axis: int = -1,
    name: Optional[str] = None,
) -> Tuple[remote_blob_util.BlobDef]:
    name = name if name is not None else id_util.UniqueStr("LayerNormGrad_")
    normalized_diff, beta_diff, gamma_diff, reduce_buf = (
        flow.user_op_builder(name)
        .Op("layer_norm_param_grad")
        .Input("dy", [dy])
        .Input("normalized", [norm])
        .Input("gamma", [gamma])
        .Output("normalized_diff")
        .Output("beta_diff")
        .Output("gamma_diff")
        .Output("reduce_buf")
        .Attr("begin_params_axis", begin_params_axis, "AttrTypeInt64")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return normalized_diff, beta_diff, gamma_diff


@oneflow_export("layers.batch_normalization")
def batch_normalization(
    inputs: remote_blob_util.BlobDef,
    axis: int = -1,
    momentum: float = 0.99,
    epsilon: float = 0.001,
    center: bool = True,
    scale: bool = True,
    beta_initializer: Optional[op_conf_util.InitializerConf] = None,
    gamma_initializer: Optional[op_conf_util.InitializerConf] = None,
    beta_regularizer: Optional[op_conf_util.RegularizerConf] = None,
    gamma_regularizer: Optional[op_conf_util.RegularizerConf] = None,
    moving_mean_initializer: Optional[op_conf_util.InitializerConf] = None,
    moving_variance_initializer: Optional[op_conf_util.InitializerConf] = None,
    trainable: bool = True,
    training: bool = True,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""
    Analogous to `tf.keras.layers.BatchNormalization <https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization>`_

    """
    assert axis >= -len(inputs.shape) and axis < len(inputs.shape)
    if axis < 0:
        axis += len(inputs.shape)
    params_shape = [inputs.shape[axis]]
    # Float32 required to avoid precision-loss when using fp16 input/output
    params_dtype = flow.float32 if inputs.dtype == flow.float16 else inputs.dtype

    if not flow.current_global_function_desc().IsTrainable() or not trainable:
        training = False

    if name is None:
        name = id_util.UniqueStr("BatchNorm_")

    if center:
        beta = flow.get_variable(
            name=name + "-beta",
            shape=params_shape,
            dtype=params_dtype,
            initializer=beta_initializer or flow.zeros_initializer(),
            regularizer=beta_regularizer,
            trainable=trainable,
            distribute=distribute_util.broadcast(),
        )
    else:
        beta = flow.constant(0, dtype=params_dtype, shape=params_shape)

    if scale:
        gamma = flow.get_variable(
            name=name + "-gamma",
            shape=params_shape,
            dtype=params_dtype,
            initializer=gamma_initializer or flow.ones_initializer(),
            regularizer=gamma_regularizer,
            trainable=trainable,
            distribute=distribute_util.broadcast(),
        )
    else:
        gamma = flow.constant(1, dtype=params_dtype, shape=params_shape)

    moving_mean = flow.get_variable(
        name=name + "-moving_mean",
        shape=params_shape,
        dtype=params_dtype,
        initializer=moving_mean_initializer or flow.zeros_initializer(),
        trainable=False,
        distribute=distribute_util.broadcast(),
    )

    moving_variance = flow.get_variable(
        name=name + "-moving_variance",
        shape=params_shape,
        dtype=params_dtype,
        initializer=moving_variance_initializer or flow.ones_initializer(),
        trainable=False,
        distribute=distribute_util.broadcast(),
    )

    builder = (
        flow.user_op_builder(name)
        .Op("normalization")
        .Input("x", [inputs])
        .Input("moving_mean", [moving_mean])
        .Input("moving_variance", [moving_variance])
        .Input("gamma", [gamma])
        .Input("beta", [beta])
        .Output("y")
        .Attr("axis", axis, "AttrTypeInt32")
        .Attr("epsilon", epsilon, "AttrTypeFloat")
        .Attr("training", training, "AttrTypeBool")
        .Attr("momentum", momentum, "AttrTypeFloat")
    )
    if trainable and training:
        builder = builder.Output("mean").Output("inv_variance")
    return builder.Build().InferAndTryRun().RemoteBlobList()[0]
