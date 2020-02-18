from __future__ import absolute_import

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("layers.dense")
def dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    trainable=True,
    name=None,
    model_distribute=distribute_util.broadcast(),
):
    in_shape = inputs.static_shape
    in_num_axes = len(in_shape)
    assert in_num_axes >= 2

    name_prefix = name if name is not None else id_util.UniqueStr("Dense_")
    inputs = (
        flow.reshape(inputs, (-1, in_shape[-1])) if in_num_axes > 2 else inputs
    )

    assert (
        model_distribute is distribute_util.auto()
        or model_distribute is distribute_util.broadcast()
        or model_distribute is distribute_util.split(0)
    )

    if model_distribute is distribute_util.split(0):
        assert (
            in_num_axes is 2
        )  # model distribute is hard for reshape split dim 1

    weight = flow.get_variable(
        name="{}-weight".format(name_prefix),
        shape=(units, inputs.static_shape[1]),
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
        a=inputs,
        b=weight,
        transpose_b=True,
        name="{}_matmul".format(name_prefix),
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
        out = flow.nn.bias_add(
            out, bias, name="{}_bias_add".format(name_prefix)
        )
    out = (
        activation(out, name="{}_activation".format(name_prefix))
        if activation is not None
        else out
    )
    out = (
        flow.reshape(out, in_shape[:-1] + (units,)) if in_num_axes > 2 else out
    )

    return out


@oneflow_export("layers.conv2d")
def conv2d(
    inputs,
    filters,
    kernel_size=1,
    strides=1,
    padding="VALID",
    data_format="NCHW",
    dilation_rate=1,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    trainable=True,
    name=None,
    weight_name=None,
    bias_name=None,
):
    name_prefix = name if name is not None else id_util.UniqueStr("Conv2D_")
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    else:
        assert isinstance(kernel_size, (list, tuple))
        kernel_size = tuple(kernel_size)
    weight_shape = (filters, inputs.static_shape[1]) + kernel_size
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
        inputs, weight, strides, padding, data_format, dilation_rate, name
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
    inputs,
    center=True,
    scale=True,
    trainable=True,
    begin_norm_axis=1,
    begin_params_axis=-1,
    epsilon=1e-5,
    name=None,
):
    op_conf = op_conf_util.OperatorConf()
    name = name if name is not None else id_util.UniqueStr("LayerNorm_")
    begin_params_axis = (
        begin_params_axis
        if begin_params_axis >= 0
        else len(inputs.shape) + begin_params_axis
    )
    param_shape = inputs.shape[begin_params_axis:]
    if len(param_shape) is 0:
        param_shape = (1,)
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
        setattr(op_conf.layer_norm_conf, "beta", beta.logical_blob_name)
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
        setattr(op_conf.layer_norm_conf, "gamma", gamma.logical_blob_name)
    setattr(op_conf, "name", name)
    setattr(op_conf, "trainable", trainable)
    setattr(op_conf.layer_norm_conf, "in", inputs.logical_blob_name)
    setattr(op_conf.layer_norm_conf, "out", "out")
    setattr(op_conf.layer_norm_conf, "center", center)
    setattr(op_conf.layer_norm_conf, "scale", scale)
    setattr(op_conf.layer_norm_conf, "begin_norm_axis", begin_norm_axis)
    setattr(op_conf.layer_norm_conf, "begin_params_axis", begin_params_axis)
    setattr(op_conf.layer_norm_conf, "epsilon", epsilon)
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)

@oneflow_export("layers.layer_norm_grad")
def layer_norm_grad(
    dy,
    x,
    mean,
    inv_variance,
    begin_norm_axis=1,
    name=None,
):
    op_conf = op_conf_util.OperatorConf()
    name = name if name is not None else id_util.UniqueStr(
        "LayerNormGrad_")
    setattr(op_conf, "name", name)
    setattr(op_conf.layer_norm_grad_conf, "dy", dy.logical_blob_name)
    setattr(op_conf.layer_norm_grad_conf, "x", x.logical_blob_name)
    setattr(op_conf.layer_norm_grad_conf, "mean", mean.logical_blob_name)
    setattr(op_conf.layer_norm_grad_conf, "inv_variance", inv_variance.logical_blob_name)
    setattr(op_conf.layer_norm_grad_conf, "dx", "dx")
    setattr(op_conf.layer_norm_grad_conf, "begin_norm_axis", begin_norm_axis)
    setattr(op_conf.layer_norm_grad_conf, "epsilon", 1e-5)
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "dx")
    return remote_blob_util.RemoteBlob(out_lbi)

@oneflow_export("layers.layer_norm_param_grad")
def layer_norm_param_grad(
    dy,
    norm,
    gamma,
    begin_params_axis=-1,
    name=None,
):
    op_conf = op_conf_util.OperatorConf()
    name = name if name is not None else id_util.UniqueStr(
        "LayerNormParamGrad_")
    setattr(op_conf, "name", name)
    setattr(op_conf.layer_norm_param_grad_conf, "dy", dy.logical_blob_name)
    setattr(op_conf.layer_norm_param_grad_conf, "normalized", norm.logical_blob_name)
    setattr(op_conf.layer_norm_param_grad_conf, "gamma", gamma.logical_blob_name)
    setattr(op_conf.layer_norm_param_grad_conf, "begin_params_axis", begin_params_axis)
    setattr(op_conf.layer_norm_param_grad_conf, "normalized_diff", "normalized_diff")
    setattr(op_conf.layer_norm_param_grad_conf, "beta_diff", "beta_diff")
    setattr(op_conf.layer_norm_param_grad_conf, "gamma_diff", "gamma_diff")
    compile_context.CurJobAddOp(op_conf)

    normalized_diff_lbi = logical_blob_id_util.LogicalBlobId()
    beta_diff_lbi = logical_blob_id_util.LogicalBlobId()
    gamma_diff_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(normalized_diff_lbi, "op_name", op_conf.name)
    setattr(beta_diff_lbi, "op_name", op_conf.name)
    setattr(gamma_diff_lbi, "op_name", op_conf.name)
    setattr(normalized_diff_lbi, "blob_name", "normalized_diff")
    setattr(beta_diff_lbi, "blob_name", "beta_diff")
    setattr(gamma_diff_lbi, "blob_name", "gamma_diff")

    return (remote_blob_util.RemoteBlob(normalized_diff_lbi),
            remote_blob_util.RemoteBlob(beta_diff_lbi),
            remote_blob_util.RemoteBlob(gamma_diff_lbi))

@oneflow_export("layers.batch_normalization")
def batch_normalization(
    inputs,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer=None,
    gamma_initializer=None,
    beta_regularizer=None,
    gamma_regularizer=None,
    moving_mean_initializer=None,
    moving_variance_initializer=None,
    trainable=True,
    name=None,
):
    assert axis >= -len(inputs.shape) and axis < len(inputs.shape)
    if axis < 0: axis += len(inputs.shape)
    params_shape = [inputs.shape[axis]]

    if name is None:
        name = id_util.UniqueStr("BatchNorm_")

    if center:
        beta = flow.get_variable(
            name=name + "-beta",
            shape=params_shape,
            dtype=inputs.dtype,
            initializer=beta_initializer or flow.zeros_initializer(),
            regularizer=beta_regularizer,
            trainable=trainable,
            distribute=distribute_util.broadcast(),
        )

    if scale:
        gamma = flow.get_variable(
            name=name + "-gamma",
            shape=params_shape,
            dtype=inputs.dtype,
            initializer=gamma_initializer or flow.ones_initializer(),
            regularizer=gamma_regularizer,
            trainable=trainable,
            distribute=distribute_util.broadcast(),
        )

    moving_mean = flow.get_variable(
        name=name + "-moving_mean",
        shape=params_shape,
        dtype=inputs.dtype,
        initializer=moving_mean_initializer or flow.zeros_initializer(),
        trainable=trainable,
        distribute=distribute_util.broadcast(),
    )

    moving_variance = flow.get_variable(
        name=name + "-moving_variance",
        shape=params_shape,
        dtype=inputs.dtype,
        initializer=moving_variance_initializer or flow.ones_initializer(),
        trainable=trainable,
        distribute=distribute_util.broadcast(),
    )

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.normalization_conf, "in", inputs.logical_blob_name)
    setattr(op_conf.normalization_conf, "out", "out")
    setattr(op_conf.normalization_conf, "axis", axis)
    setattr(op_conf.normalization_conf, "momentum", momentum)
    setattr(op_conf.normalization_conf, "epsilon", epsilon)
    setattr(op_conf.normalization_conf, "center", center)
    setattr(op_conf.normalization_conf, "scale", scale)
    setattr(
        op_conf.normalization_conf, "moving_mean", moving_mean.logical_blob_name
    )
    setattr(
        op_conf.normalization_conf,
        "moving_variance",
        moving_variance.logical_blob_name,
    )
    if beta:
        setattr(op_conf.normalization_conf, "beta", beta.logical_blob_name)
    if gamma:
        setattr(op_conf.normalization_conf, "gamma", gamma.logical_blob_name)
    if trainable:
        setattr(op_conf.normalization_conf, "mean", "mean")
        setattr(op_conf.normalization_conf, "inv_variance", "inv_variance")
        setattr(op_conf.normalization_conf, "is_training", True)
    else:
        setattr(op_conf.normalization_conf, "is_training", False)

    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("layers.batch_normalization2")
def batch_normalization2(
        inputs,
        moving_mean,
        moving_variance,
        beta=None,
        gamma=None,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        trainable=False,
        name=None,
):
    assert axis >= -len(inputs.shape) and axis < len(inputs.shape)
    params_shape = [inputs.shape[axis]]

    if name is None:
        name = id_util.UniqueStr("BatchNorm_")

    center = beta is not None
    scale = gamma is not None
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.normalization_conf, "in", inputs.logical_blob_name)
    setattr(op_conf.normalization_conf, "out", "out")
    setattr(op_conf.normalization_conf, "axis", axis)
    setattr(op_conf.normalization_conf, "momentum", momentum)
    setattr(op_conf.normalization_conf, "epsilon", epsilon)
    setattr(op_conf.normalization_conf, "center", center)
    setattr(op_conf.normalization_conf, "scale", scale)
    setattr(
        op_conf.normalization_conf, "moving_mean", moving_mean.logical_blob_name
    )
    setattr(
        op_conf.normalization_conf,
        "moving_variance",
        moving_variance.logical_blob_name,
    )
    if beta:
        setattr(op_conf.normalization_conf, "beta", beta.logical_blob_name)
    if gamma:
        setattr(op_conf.normalization_conf, "gamma", gamma.logical_blob_name)
    if trainable:
        setattr(op_conf.normalization_conf, "mean", "mean")
        setattr(op_conf.normalization_conf, "inv_variance", "inv_variance")
        setattr(op_conf.normalization_conf, "is_training", True)
    else:
        setattr(op_conf.normalization_conf, "is_training", False)

    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)
