from __future__ import absolute_import

import os

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
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
        assert in_num_axes is 2  # model distribute is hard for reshape split dim 1

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
    inputs,
    filters,
    kernel_size=1,
    strides=1,
    padding="VALID",
    data_format="NCHW",
    dilation_rate=1,
    groups=1,
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
    inputs,
    center=True,
    scale=True,
    trainable=True,
    begin_norm_axis=1,
    begin_params_axis=-1,
    epsilon=1e-5,
    name=None,
):
    r"""
    Analogous to `tf.keras.layers.LayerNormalization <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization>`_

    """
    if os.getenv("ENABLE_USER_OP") == "True":
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
    else:
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
            setattr(op_conf.layer_norm_conf, "beta", beta.unique_name)
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
            setattr(op_conf.layer_norm_conf, "gamma", gamma.unique_name)
        setattr(op_conf, "name", name)
        setattr(op_conf, "trainable", trainable)
        setattr(op_conf.layer_norm_conf, "in", inputs.unique_name)
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
    dy, x, mean, inv_variance, begin_norm_axis=1, name=None,
):
    op_conf = op_conf_util.OperatorConf()
    name = name if name is not None else id_util.UniqueStr("LayerNormGrad_")
    setattr(op_conf, "name", name)
    setattr(op_conf.layer_norm_grad_conf, "dy", dy.unique_name)
    setattr(op_conf.layer_norm_grad_conf, "x", x.unique_name)
    setattr(op_conf.layer_norm_grad_conf, "mean", mean.unique_name)
    setattr(op_conf.layer_norm_grad_conf, "inv_variance", inv_variance.unique_name)
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
    dy, norm, gamma, begin_params_axis=-1, name=None,
):
    op_conf = op_conf_util.OperatorConf()
    name = name if name is not None else id_util.UniqueStr("LayerNormParamGrad_")
    setattr(op_conf, "name", name)
    setattr(op_conf.layer_norm_param_grad_conf, "dy", dy.unique_name)
    setattr(op_conf.layer_norm_param_grad_conf, "normalized", norm.unique_name)
    setattr(op_conf.layer_norm_param_grad_conf, "gamma", gamma.unique_name)
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

    return (
        remote_blob_util.RemoteBlob(normalized_diff_lbi),
        remote_blob_util.RemoteBlob(beta_diff_lbi),
        remote_blob_util.RemoteBlob(gamma_diff_lbi),
    )


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
    training=True,
    name=None,
):
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

    if os.getenv("ENABLE_USER_OP") == "True":
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

    else:
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

        moving_mean = flow.get_variable(
            name=name + "-moving_mean",
            shape=params_shape,
            dtype=params_dtype,
            initializer=moving_mean_initializer or flow.zeros_initializer(),
            trainable=trainable,
            distribute=distribute_util.broadcast(),
        )

        moving_variance = flow.get_variable(
            name=name + "-moving_variance",
            shape=params_shape,
            dtype=params_dtype,
            initializer=moving_variance_initializer or flow.ones_initializer(),
            trainable=trainable,
            distribute=distribute_util.broadcast(),
        )

        op_conf = op_conf_util.OperatorConf()
        setattr(op_conf, "name", name)
        setattr(op_conf.normalization_conf, "in", inputs.unique_name)
        setattr(op_conf.normalization_conf, "out", "out")
        setattr(op_conf.normalization_conf, "axis", axis)
        setattr(op_conf.normalization_conf, "momentum", momentum)
        setattr(op_conf.normalization_conf, "epsilon", epsilon)
        setattr(op_conf.normalization_conf, "center", center)
        setattr(op_conf.normalization_conf, "scale", scale)
        setattr(op_conf.normalization_conf, "moving_mean", moving_mean.unique_name)
        setattr(
            op_conf.normalization_conf, "moving_variance", moving_variance.unique_name,
        )
        if center:
            setattr(op_conf.normalization_conf, "beta", beta.unique_name)
        if scale:
            setattr(op_conf.normalization_conf, "gamma", gamma.unique_name)
        if trainable and flow.current_global_function_desc().IsTrainable():
            if not training:
                raise ValueError(
                    "training == False && trainable == True doesn't work in non-user-op mode"
                )
            setattr(op_conf.normalization_conf, "mean", "mean")
            setattr(op_conf.normalization_conf, "inv_variance", "inv_variance")
            setattr(op_conf.normalization_conf, "is_training", training)
        else:
            setattr(op_conf.normalization_conf, "is_training", False)

        compile_context.CurJobAddOp(op_conf)
        out_lbi = logical_blob_id_util.LogicalBlobId()
        setattr(out_lbi, "op_name", op_conf.name)
        setattr(out_lbi, "blob_name", "out")
        return remote_blob_util.RemoteBlob(out_lbi)
