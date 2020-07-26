"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from typing import Callable, Optional, Union, Tuple, Sequence
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.remote_blob as remote_blob_util


@oneflow_export("layers.dense")
def dense(
    inputs: remote_blob_util.BlobDef,
    units: int,
    activation: Optional[
        Callable[[remote_blob_util.BlobDef, str], remote_blob_util.BlobDef]
    ] = None,
    use_bias: bool = True,
    kernel_initializer: Optional[op_conf_util.InitializerConf] = None,
    bias_initializer: Optional[op_conf_util.InitializerConf] = None,
    kernel_regularizer: Optional[op_conf_util.RegularizerConf] = None,
    bias_regularizer: Optional[op_conf_util.RegularizerConf] = None,
    trainable: bool = True,
    name: str = "Dense",
    model_distribute: distribute_util.Distribute = distribute_util.broadcast(),
) -> remote_blob_util.BlobDef:
    r"""
    Analogous to `tf.keras.layers.Dense <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`_

    """
    in_shape = inputs.shape
    in_num_axes = len(in_shape)
    assert in_num_axes >= 2

    assert (
        model_distribute is distribute_util.auto()
        or model_distribute is distribute_util.broadcast()
        or model_distribute is distribute_util.split(0)
    )

    if model_distribute is distribute_util.split(0):
        assert in_num_axes == 2  # model distribute is hard for reshape split dim 1

    if in_num_axes > 2:
        inputs = flow.reshape(inputs, (-1, in_shape[-1]))

    with flow.scope.namespace(name):
        if kernel_initializer is None:
            kernel_initializer = flow.constant_initializer(0)

        weight = flow.get_variable(
            name="weight",
            shape=(units, inputs.shape[1]),
            dtype=inputs.dtype,
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
            trainable=trainable,
            model_name="weight",
            distribute=model_distribute,
            reuse=False,
        )
        weight = weight.with_distribute(model_distribute)

        out = flow.matmul(a=inputs, b=weight, transpose_b=True, name="matmul")

        if use_bias:
            if bias_initializer is None:
                bias_initializer = flow.constant_initializer(0)

            bias = flow.get_variable(
                name="bias",
                shape=(units,),
                dtype=inputs.dtype,
                initializer=bias_initializer,
                regularizer=bias_regularizer,
                trainable=trainable,
                model_name="bias",
                distribute=model_distribute,
                reuse=False,
            )
            bias = bias.with_distribute(model_distribute)
            out = flow.nn.bias_add(out, bias, name="bias_add")

        if callable(activation):
            out = activation(out, name="activation")

    if in_num_axes > 2:
        out = flow.reshape(out, in_shape[:-1] + (units,))

    return out


@oneflow_export("layers.conv2d")
def conv2d(
    inputs: remote_blob_util.BlobDef,
    filters: int,
    kernel_size: Union[int, Sequence[int]] = 1,
    strides: Union[int, Sequence[int]] = 1,
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
    name: str = "Conv2d",
    weight_name: Optional[str] = None,
    bias_name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    else:
        assert isinstance(kernel_size, (list, tuple))
        assert len(kernel_size) == 2
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

    if kernel_initializer is None:
        kernel_initializer = flow.constant_initializer(0)

    if weight_name is None:
        with flow.scope.namespace(name):
            weight = flow.get_variable(
                name="weight",
                shape=weight_shape,
                dtype=inputs.dtype,
                initializer=kernel_initializer,
                regularizer=kernel_regularizer,
                trainable=trainable,
                model_name="weight",
                reuse=False,
            )
    else:
        weight = flow.get_variable(
            name=weight_name,
            shape=weight_shape,
            dtype=inputs.dtype,
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
            trainable=trainable,
            model_name="weight",
            reuse=False,
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
        if bias_initializer is None:
            bias_initializer = flow.constant_initializer(0)

        if bias_name is None:
            with flow.scope.namespace(name):
                bias = flow.get_variable(
                    name="bias",
                    shape=(filters,),
                    dtype=inputs.dtype,
                    initializer=bias_initializer,
                    regularizer=bias_regularizer,
                    trainable=trainable,
                    model_name="bias",
                    reuse=False,
                )
        else:
            bias = flow.get_variable(
                name=bias_name,
                shape=(filters,),
                dtype=inputs.dtype,
                initializer=bias_initializer,
                regularizer=bias_regularizer,
                trainable=trainable,
                model_name="bias",
                reuse=False,
            )

        with flow.scope.namespace(name):
            output = flow.nn.bias_add(output, bias, data_format, name="bias_add")

    if callable(activation):
        with flow.scope.namespace(name):
            output = activation(output, name="activation")

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
    name: str = "LayerNorm",
) -> remote_blob_util.BlobDef:
    r"""
    Analogous to `tf.keras.layers.LayerNormalization <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization>`_

    """
    op_builder = (
        flow.user_op_builder(name)
        .Op("layer_norm")
        .Input("x", [inputs])
        .Output("y")
        .Output("mean")
        .Output("inv_variance")
    )

    if center is False and scale is False:
        trainable = False

    param_shape = inputs.shape[begin_params_axis:]
    if center:
        with flow.scope.namespace(name):
            beta = flow.get_variable(
                name="beta",
                shape=param_shape,
                dtype=inputs.dtype,
                initializer=flow.constant_initializer(0.0),
                trainable=trainable,
                model_name="beta",
                distribute=distribute_util.broadcast(),
                reuse=False,
            )

        op_builder.Input("beta", [beta])

    if scale:
        with flow.scope.namespace(name):
            gamma = flow.get_variable(
                name="gamma",
                shape=param_shape,
                dtype=inputs.dtype,
                initializer=flow.constant_initializer(1.0),
                trainable=trainable,
                model_name="gamma",
                distribute=distribute_util.broadcast(),
                reuse=False,
            )

        op_builder.Input("gamma", [gamma])
        op_builder.Output("normalized")

    op_builder.Attr("center", center)
    op_builder.Attr("scale", scale)
    op_builder.Attr("begin_norm_axis", begin_norm_axis)
    op_builder.Attr("begin_params_axis", begin_params_axis)
    op_builder.Attr("epsilon", epsilon)

    return op_builder.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("layers.layer_norm_grad")
def layer_norm_grad(
    dy: remote_blob_util.BlobDef,
    x: remote_blob_util.BlobDef,
    mean: remote_blob_util.BlobDef,
    inv_variance: remote_blob_util.BlobDef,
    begin_norm_axis: int = 1,
    name: str = "LayerNormGrad",
) -> remote_blob_util.BlobDef:
    op = (
        flow.user_op_builder(name)
        .Op("layer_norm_grad")
        .Input("dy", [dy])
        .Input("x", [x])
        .Input("mean", [mean])
        .Input("inv_variance", [inv_variance])
        .Output("dx")
        .Attr("begin_norm_axis", begin_norm_axis)
        .Attr("epsilon", 1e-5)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("layers.layer_norm_param_grad")
def layer_norm_param_grad(
    dy: remote_blob_util.BlobDef,
    norm: remote_blob_util.BlobDef,
    gamma: remote_blob_util.BlobDef,
    begin_params_axis: int = -1,
    name: str = "LayerNormParamGrad",
) -> Tuple[
    remote_blob_util.BlobDef, remote_blob_util.BlobDef, remote_blob_util.BlobDef
]:
    op = (
        flow.user_op_builder(name)
        .Op("layer_norm_param_grad")
        .Input("dy", [dy])
        .Input("normalized", [norm])
        .Input("gamma", [gamma])
        .Output("normalized_diff")
        .Output("beta_diff")
        .Output("gamma_diff")
        .Output("reduce_buf")
        .Attr("begin_params_axis", begin_params_axis)
        .Build()
    )

    (
        normalized_diff,
        beta_diff,
        gamma_diff,
        reduce_buf,
    ) = op.InferAndTryRun().RemoteBlobList()

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
    name: str = "BatchNorm",
) -> remote_blob_util.BlobDef:
    r"""
    Analogous to `tf.keras.layers.BatchNormalization <https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization>`_

    """
    if axis < 0:
        axis += len(inputs.shape)
    assert axis >= 0 and axis < len(inputs.shape)

    params_shape = [inputs.shape[axis]]
    # Float32 required to avoid precision-loss when using fp16 input/output
    params_dtype = flow.float32 if inputs.dtype == flow.float16 else inputs.dtype

    if not flow.current_global_function_desc().IsTrainable() or not trainable:
        training = False

    with flow.scope.namespace(name):
        if center:
            beta = flow.get_variable(
                name="beta",
                shape=params_shape,
                dtype=params_dtype,
                initializer=beta_initializer or flow.zeros_initializer(),
                regularizer=beta_regularizer,
                trainable=trainable,
                distribute=distribute_util.broadcast(),
                reuse=False,
            )
        else:
            beta = flow.constant(0, dtype=params_dtype, shape=params_shape, name="beta")

        if scale:
            gamma = flow.get_variable(
                name="gamma",
                shape=params_shape,
                dtype=params_dtype,
                initializer=gamma_initializer or flow.ones_initializer(),
                regularizer=gamma_regularizer,
                trainable=trainable,
                distribute=distribute_util.broadcast(),
                reuse=False,
            )
        else:
            gamma = flow.constant(
                1, dtype=params_dtype, shape=params_shape, name="gamma"
            )

        moving_mean = flow.get_variable(
            name="moving_mean",
            shape=params_shape,
            dtype=params_dtype,
            initializer=moving_mean_initializer or flow.zeros_initializer(),
            trainable=False,
            distribute=distribute_util.broadcast(),
            reuse=False,
        )

        moving_variance = flow.get_variable(
            name="moving_variance",
            shape=params_shape,
            dtype=params_dtype,
            initializer=moving_variance_initializer or flow.ones_initializer(),
            trainable=False,
            distribute=distribute_util.broadcast(),
            reuse=False,
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
        .Attr("axis", axis)
        .Attr("epsilon", epsilon)
        .Attr("training", training)
        .Attr("momentum", momentum)
    )
    if trainable and training:
        builder = builder.Output("mean").Output("inv_variance")

    return builder.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("layers.upsample_2d")
def upsample(
    x: remote_blob_util.BlobDef,
    size: Sequence[int] = (2, 2),
    data_format: str = "NCHW",
    interpolation: str = "nearest",
    name: str = "Upsample2D",
):
    if isinstance(size, int):
        height_scale = size
        width_scale = size
    else:
        assert isinstance(size, (list, tuple))
        assert len(size) == 2
        height_scale = size[0]
        width_scale = size[1]

    if interpolation != "nearest" and interpolation != "bilinear":
        raise ValueError('interpolation must be "nearest" or "bilinear".')

    if data_format.upper() != "NCHW" and data_format.upper() != "NHWC":
        raise ValueError('data_format must be "NHWC" or "NCHW".')

    need_transpose = 0
    if data_format.upper() == "NHWC":
        need_transpose = 1

    if need_transpose:
        x = flow.transpose(x, perm=[0, 3, 1, 2])

    op = (
        flow.user_op_builder(name)
        .Op("upsample")
        .Input("x", [x])
        .Output("y")
        .Attr("height_scale", float(height_scale))
        .Attr("width_scale", float(width_scale))
        .Attr("data_format", "channels_first")
        .Attr("interpolation", interpolation)
        .Build()
    )
    output = op.InferAndTryRun().SoleOutputBlob()

    if need_transpose:
        output = flow.transpose(output, perm=[0, 2, 3, 1])

    return output
