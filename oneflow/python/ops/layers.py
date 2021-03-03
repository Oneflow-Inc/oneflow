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
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util
import oneflow.core.job.regularizer_conf_pb2 as regularizer_conf_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow_api

IntPair = Tuple[int, int]


@oneflow_export("layers.dense")
def dense(
    inputs: oneflow_api.BlobDesc,
    units: int,
    activation: Optional[
        Callable[[oneflow_api.BlobDesc, str], oneflow_api.BlobDesc]
    ] = None,
    use_bias: bool = True,
    kernel_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    bias_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    kernel_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    bias_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    trainable: bool = True,
    name: str = "Dense",
    model_distribute: oneflow_api.sbp.Sbp = oneflow_api.sbp.broadcast(),
) -> oneflow_api.BlobDesc:
    r"""Fully-connected layer. 

    The fully-connected layer multiplies input Blob with weight matrix and produces an Output Blob. 

    Args:
        inputs (oneflow_api.BlobDesc): A 2D input `Blob`.
        units (int): A positive integer for the dimensionality of the output space.
        activation (Optional[oneflow_api.BlobDesc], optional):  Activation function. Defaults to None.
        use_bias (bool, optional): A boolean specifies whether to use a bias vector. Defaults to True.
        kernel_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for the kernel weights matrix. Defaults to None.
        bias_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for the bias vector. Defaults to None.
        kernel_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer function applied to the kernel weights matrix. Defaults to None.
        bias_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for the bias vector. Defaults to None.
        trainable (bool, optional): A boolean specifies whether to train the variables. Defaults to True.
        name (Optional[str], optional): This layer's name. Defaults to None.
        model_distribute (oneflow_api.sbp.Sbp, optional): Define the way to ditribute the model. Defaults to oneflow_api.sbp.broadcast().

    Returns:
        oneflow_api.BlobDesc:  A N-D `Blob` with the shape of (batch_size, units).

    Raises:
        ValueError: The dimension of input `Blob` must be less than 2.
        VauleError: Model distribute must be in auto, broadcast, split.
        ValueError: The input must be a 2D `Blob` when the model distribute is split.

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def dense_Job(x: tp.Numpy.Placeholder((1, 256))
        ) -> tp.Numpy:
            initializer = flow.truncated_normal(0.1)
            hidden = flow.layers.dense(
                x,
                512,
                activation=flow.nn.relu,
                kernel_initializer=initializer,
                name="dense1",
            )
            return hidden


        x = np.random.randn(1, 256).astype(np.float32)
        out = dense_Job(x)

        # out.shape (1, 512)

    """
    in_shape = inputs.shape
    in_num_axes = len(in_shape)
    assert in_num_axes >= 2

    assert (
        model_distribute is oneflow_api.sbp.auto()
        or model_distribute is oneflow_api.sbp.broadcast()
        or model_distribute is oneflow_api.sbp.split(0)
    )

    if model_distribute is oneflow_api.sbp.split(0):
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


@oneflow_export("layers.conv1d")
def conv1d(
    inputs: oneflow_api.BlobDesc,
    filters: int,
    kernel_size: Union[int, Tuple[int]] = 1,
    strides: Union[int, Tuple[int]] = 1,
    padding: Union[str, Tuple[IntPair, IntPair, IntPair]] = "VALID",
    data_format: str = "NCW",
    dilation_rate: Optional[Union[int, Tuple[int]]] = None,
    groups: int = 1,
    activation: Optional[
        Callable[[oneflow_api.BlobDesc, str], oneflow_api.BlobDesc]
    ] = None,
    use_bias: bool = True,
    kernel_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    bias_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    kernel_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    bias_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    trainable: bool = True,
    name: str = "Conv1d",
    weight_name: Optional[str] = None,
    bias_name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""1D convolution layer. 

    This layer computes a 1-D convolution with 3D input Blob and filters. 

    Args:
        inputs (oneflow_api.BlobDesc): A 3D input `Blob`.
        filters (int): An integer specifies the dimensionality of the output space.
        kernel_size (Union[int, List[int], Tuple[int]], optional): An integer or tuple/list specifies the height and width of the convolution window.
                        When it is an integer, a square window is applied to the input. Defaults to 1.
        strides (Union[int, List[int], Tuple[int]], optional): An integer or tuple/list specifies the strides of the convolution window along the height and width.
                        When it is an integer, the same value for the all spatial dimesions is applied. Defaults to 1.
        padding (str, Tuple[IntPair, IntPair, IntPair], optional): padding: `string` `"SAME"` or `"SAME_LOWER"` or `"SAME_UPPER"` or `"VALID" or Tuple[IntPair, IntPair, IntPair]` indicating the type of padding algorithm to use, or a list indicating the explicit paddings at the start and end of each dimension. Defaults to "VALID".
        data_format (str, optional): A string specifies the format of the input `Blob`, one of "NCW" or "NWC" (default: "NCW"). "NCW" cooresponds to channels_first, i.e. the input `Blob` with shape (batch_size, channels, width).
                        "NWC" cooresponds to channels_last, i.e. the input `Blob` with shape (batch_size, channels, width). Defaults to "NCW".
        dilation_rate (Optional[Union[int, Tuple[int]]], optional): An integer or tuple/list specifies the dilation rate for the dilated convolution. When it is an integer, the same dilation rate is applied for the all dimensions. Defaults to 1.
        groups (int, optional): A positive integer specifies number of groups for the Group conv. Defaults to 1.
        activation (Optional[ Callable[[oneflow_api.BlobDesc, str], oneflow_api.BlobDesc] ], optional): Activation function. Defaults to None.
        use_bias (bool, optional): A boolean specifies whether to use a bias vector. Defaults to True.
        kernel_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for the kernel weights matrix. Defaults to None.
        bias_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for the bias vector. Defaults to None.
        kernel_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for the kernel weights matrix. Defaults to None.
        bias_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for the bias vector . Defaults to None.
        trainable (bool, optional): A boolean specifies whether to train variables. Defaults to True.
        name (Optional[str], optional): This layer's name. Defaults to None.

    Raises:
        ValueError: If the type of kernel_size is not one of integer, list, tuple.
        ValueError: The number of groups must be positive and number of filters must be divisible by it.
        ValueError: If data_format is not one of 'NCW', 'NWC'.
        ValueError: If number of input channels is not divisible by number of groups or less than number of groups.
        ValueError: Number of group must be one when data_format is 'NWC'.

    Returns:
        oneflow_api.BlobDesc: A 3D `Blob` with the shape of (batch_size, filters, new_width).

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv1d_Job(x: tp.Numpy.Placeholder((1, 64, 32))
        ) -> tp.Numpy:
            initializer = flow.truncated_normal(0.1)
            conv1d = flow.layers.conv1d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer,
                name="Conv1d"
            )
            return conv1d


        x = np.random.randn(1, 64, 32).astype(np.float32)
        out = conv1d_Job(x)

        # out.shape (1, 128, 32)

    """

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    else:
        assert isinstance(kernel_size, (list, tuple))
        assert len(kernel_size) == 1
        kernel_size = tuple(kernel_size)

    assert isinstance(groups, int)
    assert groups > 0
    assert groups <= filters
    assert filters % groups == 0

    if data_format.upper() == "NCW":
        assert groups <= inputs.shape[1]
        assert inputs.shape[1] % groups == 0
        weight_shape = (filters, inputs.shape[1] // groups) + kernel_size
    elif data_format.upper() == "NWC":
        assert groups == 1
        assert groups <= inputs.shape[2]
        assert inputs.shape[2] % groups == 0
        weight_shape = (
            filters,
            kernel_size[0],
            inputs.shape[2] // groups,
        )
    else:
        raise ValueError("data_format must be in NCW or NWC")

    if kernel_initializer is None:
        kernel_initializer = flow.xavier_uniform_initializer(data_format=data_format)

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

    output = flow.nn.conv1d(
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


@oneflow_export("layers.conv2d")
def conv2d(
    inputs: oneflow_api.BlobDesc,
    filters: int,
    kernel_size: Union[int, IntPair] = 1,
    strides: Union[int, IntPair] = 1,
    padding: Union[str, Tuple[IntPair, IntPair, IntPair, IntPair]] = "VALID",
    data_format: str = "NCHW",
    dilation_rate: Optional[Union[int, IntPair]] = None,
    groups: int = 1,
    activation: Optional[
        Callable[[oneflow_api.BlobDesc, str], oneflow_api.BlobDesc]
    ] = None,
    use_bias: bool = True,
    kernel_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    bias_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    kernel_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    bias_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    trainable: bool = True,
    name: str = "Conv2d",
    weight_name: Optional[str] = None,
    bias_name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""2D convolution layer. 

    This layer computes a 2D convolution with 4D input Blob and filters. 

    Args:
        inputs (oneflow_api.BlobDesc): A 4D input `Blob`.
        filters (int): An integer specifies the dimensionality of the output space.
        kernel_size (Union[int, List[int], Tuple[int]], optional): An integer or tuple/list specifies the height and width of the convolution window.
                        When it is an integer, a square window is applied to the input. Defaults to 1.
        strides (Union[int, List[int], Tuple[int]], optional): An integer or tuple/list specifies the strides of the convolution window along the height and width.
                        When it is an integer, the same value for the all spatial dimesions is applied. Defaults to 1.
        padding (str, Tuple[IntPair, IntPair, IntPair, IntPair], optional): padding: `string` `"SAME"` or `"SAME_LOWER"` or `"SAME_UPPER"` or `"VALID" or Tuple[IntPair, IntPair, IntPair]` indicating the type of padding algorithm to use, or a list indicating the explicit paddings at the start and end of each dimension. Defaults to "VALID".
        data_format (str, optional): A string specifies the format of the input `Blob`, one of "NCHW" or "NHWC" (default: "NCHW"). "NCHW" cooresponds to channels_first, i.e. the input `Blob` with shape (batch_size, channels, height, width).
                        "NHWC" cooresponds to channels_last, i.e. the input `Blob` with shape (batch_size, height, width, channels). Defaults to "NCHW".
        dilation_rate (int, optional): An integer or tuple/list specifies the dilation rate for the dilated convolution. When it is an integer, the same dilation rate is applied for the all dimensions. Defaults to 1.
        groups (int, optional): A positive integer specifies number of groups for the Group conv. Defaults to 1.
        activation (Optional[ Callable[[oneflow_api.BlobDesc, str], oneflow_api.BlobDesc] ], optional): Activation function. Defaults to None.
        use_bias (bool, optional): A boolean specifies whether to use a bias vector. Defaults to True.
        kernel_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for the kernel weights matrix. Defaults to None.
        bias_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for the bias vector. Defaults to None.
        kernel_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for the kernel weights matrix. Defaults to None.
        bias_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for the bias vector . Defaults to None.
        trainable (bool, optional): A boolean specifies whether to train variables. Defaults to True.
        name (Optional[str], optional): This layer's name. Defaults to None.
        weight_name (Optional[str], optional): This weight's name. Defaults to None.
        bias_name (Optional[str], optional):  This bias's name. Defaults to None.

    Raises:
        ValueError: If the type of kernel_size is not one of integer, list, tuple.
        ValueError: The number of groups must be positive and number of filters must be divisible by it.
        ValueError: If data_format is not one of 'NCHW', 'NHWC'.
        ValueError: If number of input channels is not divisible by number of groups or less than number of groups.
        ValueError: Number of group must be one when data_format is 'NHWC'.

    Returns:
        oneflow_api.BlobDesc: A 4D `Blob` with the shape of (batch_size, filters, new_height, new_width).

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.truncated_normal(0.1)
            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer,
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_Job(x)

        # out.shape (1, 128, 32, 32)

    """

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
        kernel_initializer = flow.xavier_uniform_initializer(data_format=data_format)

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


@oneflow_export("layers.conv3d")
def conv3d(
    inputs: oneflow_api.BlobDesc,
    filters: int,
    kernel_size: Union[int, Sequence[int]] = 1,
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, Tuple[IntPair, IntPair, IntPair, IntPair, IntPair]] = "VALID",
    data_format: str = "NCDHW",
    dilation_rate: Optional[Union[int, IntPair]] = None,
    groups: int = 1,
    activation: Optional[
        Callable[[oneflow_api.BlobDesc, str], oneflow_api.BlobDesc]
    ] = None,
    use_bias: bool = True,
    kernel_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    bias_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    kernel_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    bias_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    trainable: bool = True,
    name: str = "Conv3d",
    weight_name: Optional[str] = None,
    bias_name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""3D convolution layer. 

    This layer computes 3D convolution with 5D input Blob and filters

    Args:
        inputs (oneflow_api.BlobDesc): A 5D input `Blob`.
        filters (int): An integer specifies the dimensionality of the output space.
        kernel_size (Union[int, List[int], Sequence[int]], optional): An integer or tuple/list specifies the height and width of the convolution window.
                        When it is an integer, a square window is applied to the input. Defaults to 1.
        strides (Union[int, List[int], Sequence[int]], optional): An integer or tuple/list specifies the strides of the convolution window along the height and width.
                        When it is an integer, the same value for the all spatial dimesions is applied. Defaults to 1.
        padding (str, Tuple[IntPair, IntPair, IntPair, IntPair, IntPair], optional): padding: `string` `"SAME"` or `"SAME_LOWER"` or `"SAME_UPPER"` or `"VALID" or Tuple[IntPair, IntPair, IntPair, IntPair, IntPair]` indicating the type of padding algorithm to use, or a list indicating the explicit paddings at the start and end of each dimension. Defaults to "VALID".
        data_format (str, optional): A string specifies the format of the input `Blob`, one of "NCDHW" or "NDHWC" (default: "NCDHW"). "NCDHW" cooresponds to channels_first, i.e. the input `Blob` with shape (batch_size, channels, depth, height, width).
                        "NDHWC" cooresponds to channels_last, i.e. the input `Blob` with shape (batch_size, channels, depth, height, width). Defaults to "NCDHW".
        dilation_rate (int, optional): An integer or tuple/list specifies the dilation rate for the dilated convolution. When it is an integer, the same dilation rate is applied for the all dimensions. Defaults to 1.
        groups (int, optional): A positive integer specifies number of groups for the Group conv. Defaults to 1.
        activation (Optional[ Callable[[oneflow_api.BlobDesc, str], oneflow_api.BlobDesc] ], optional): Activation function. Defaults to None.
        use_bias (bool, optional): A boolean specifies whether to use a bias vector. Defaults to True.
        kernel_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for the kernel weights matrix. Defaults to None.
        bias_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for the bias vector. Defaults to None.
        kernel_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for the kernel weights matrix. Defaults to None.
        bias_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for the bias vector . Defaults to None.
        trainable (bool, optional): A boolean specifies whether to train variables. Defaults to True.
        name (Optional[str], optional): This layer's name. Defaults to None.
        weight_name (Optional[str], optional): This weight's name. Defaults to None.
        bias_name (Optional[str], optional):  This bias's name. Defaults to None.

    Raises:
        ValueError: If the type of kernel_size is not one of integer, list, tuple.
        ValueError: The number of groups must be positive and number of filters must be divisible by it.
        ValueError: If data_format is not one of 'NCDHW', 'NDHWC'.
        ValueError: If number of input channels is not divisible by number of groups or less than number of groups.
        ValueError: Number of group must be one when data_format is 'NDHWC'.

    Returns:
        oneflow_api.BlobDesc: A 5D `Blob` with the shape of (batch_size, filters, new_height, new_width).

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv3d_Job(x: tp.Numpy.Placeholder((1, 64, 16, 16, 16))
        ) -> tp.Numpy:
            initializer = flow.truncated_normal(0.1)
            conv3d = flow.layers.conv3d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer,
                name="Conv3d"
            )
            return conv3d


        x = np.random.randn(1, 64, 16, 16, 16).astype(np.float32)
        out = conv3d_Job(x)

        # out.shape (1, 128, 16, 16, 16)

    """
    need_transpose = 0
    if data_format.upper() == "NDHWC":  # NDHWC is not supported before cudnn 8.0
        need_transpose = 1
        data_format = "NCDHW"

    if need_transpose:
        inputs = flow.transpose(inputs, perm=[0, 4, 1, 2, 3])
        # padding for `NDHWC` is [0, 0, 1, 1, 1] to `NCDHW` format [0, 1, 1, 1, 0]
        if isinstance(padding, (list, tuple)):
            padding = list(padding)
            padding[1], padding[4] = padding[4], padding[1]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    else:
        assert isinstance(kernel_size, (list, tuple))
        assert len(kernel_size) == 3
        kernel_size = tuple(kernel_size)

    assert isinstance(groups, int)
    assert groups > 0
    assert groups <= filters
    assert filters % groups == 0

    if data_format.upper() == "NCDHW":
        assert groups <= inputs.shape[1]
        assert inputs.shape[1] % groups == 0
        weight_shape = (filters, inputs.shape[1] // groups) + kernel_size
    elif data_format.upper() == "NDHWC":
        assert groups == 1
        assert groups <= inputs.shape[3]
        assert inputs.shape[3] % groups == 0
        weight_shape = (
            filters,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
            inputs.shape[4] // groups,
        )
    else:
        raise ValueError("data_format must be in NCHW or NHWC")

    if kernel_initializer is None:
        kernel_initializer = flow.xavier_uniform_initializer(data_format=data_format)

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

    output = flow.nn.conv3d(
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

    if need_transpose:
        output = flow.transpose(output, perm=[0, 2, 3, 4, 1])

    return output


@oneflow_export("layers.layer_norm")
def layer_norm(
    inputs: oneflow_api.BlobDesc,
    center: bool = True,
    scale: bool = True,
    trainable: bool = True,
    begin_norm_axis: int = 1,
    begin_params_axis: int = -1,
    epsilon: float = 1e-5,
    name: str = "LayerNorm",
) -> oneflow_api.BlobDesc:
    r"""Layer Normalization. 

    Args:
        inputs (oneflow_api.BlobDesc): Input `Blob`.
        center (bool, optional): A boolean specifies whether to shift input `Blob`. Defaults to True.
        scale (bool, optional): A boolean specifies whether to scale input `Blob`. Defaults to True.
        trainable (bool, optional): A boolean specifies whether to train variables. Defaults to True.
        begin_norm_axis (int, optional): An integer specifies which axis to normalize at first. Defaults to 1.
        begin_params_axis (int, optional):  An integer specifies which axis params at . Defaults to -1.
        epsilon (float, optional): A small float is added to avoid division by zero. Defaults to 1e-5.
        name (Optional[str], optional): This layer's name. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A normalized `Blob` with same shape of input.

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def layer_norm_Job(x: tp.Numpy.Placeholder((1, 64, 128, 128))
        ) -> tp.Numpy:
            layer_norm = flow.layers.layer_norm(
                x,
                name="LayerNorm1"
            )
            return layer_norm


        x = np.random.randn(1, 64, 128, 128).astype(np.float32)
        out = layer_norm_Job(x)

        # out.shape (1, 64, 128, 128)

    """
    if center is False and scale is False:
        trainable = False

    beta = None
    gamma = None

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
                distribute=oneflow_api.sbp.broadcast(),
                reuse=False,
            )

    if scale:
        with flow.scope.namespace(name):
            gamma = flow.get_variable(
                name="gamma",
                shape=param_shape,
                dtype=inputs.dtype,
                initializer=flow.constant_initializer(1.0),
                trainable=trainable,
                model_name="gamma",
                distribute=oneflow_api.sbp.broadcast(),
                reuse=False,
            )

    if flow.current_scope().device_parallel_desc_symbol.device_tag == "cpu":
        if begin_norm_axis < 0:
            begin_norm_axis = begin_norm_axis + len(inputs.shape)

        reduce_axis = []
        for dim in range(len(inputs.shape)):
            if dim >= begin_norm_axis:
                reduce_axis.append(dim)
        mean, variance = flow.nn.moments(inputs, reduce_axis, keepdims=True)

        axis = begin_norm_axis
        normalized = flow.nn.batch_normalization(
            x=inputs,
            mean=mean,
            variance=variance,
            variance_epsilon=epsilon,
            axis=axis,
            name=name,
        )
        nd_params_shape = [1] * (len(inputs.shape) - len(param_shape)) + list(
            param_shape
        )
        affined = normalized
        if gamma:
            gamma = flow.reshape(gamma, nd_params_shape)
            affined *= gamma
        if beta:
            beta = flow.reshape(beta, nd_params_shape)
            affined += beta
        return affined
    elif flow.current_scope().device_parallel_desc_symbol.device_tag == "gpu":
        op_builder = (
            flow.user_op_builder(name)
            .Op("layer_norm")
            .Input("x", [inputs])
            .Output("y")
            .Output("mean")
            .Output("inv_variance")
        )

        if beta is not None:
            op_builder.Input("beta", [beta])
        if gamma is not None:
            op_builder.Input("gamma", [gamma])
            op_builder.Output("normalized")
        op_builder.Attr("center", center)
        op_builder.Attr("scale", scale)
        op_builder.Attr("begin_norm_axis", begin_norm_axis)
        op_builder.Attr("begin_params_axis", begin_params_axis)
        op_builder.Attr("epsilon", epsilon)

        return op_builder.Build().InferAndTryRun().RemoteBlobList()[0]
    else:
        raise NotImplementedError


@oneflow_export("layers.layer_norm_grad")
def layer_norm_grad(
    dy: oneflow_api.BlobDesc,
    x: oneflow_api.BlobDesc,
    mean: oneflow_api.BlobDesc,
    inv_variance: oneflow_api.BlobDesc,
    begin_norm_axis: int = 1,
    name: str = "LayerNormGrad",
) -> oneflow_api.BlobDesc:
    r"""Layer normalization

    Args:
        dy (oneflow_api.BlobDesc): Upstream derivstives.
        x (oneflow_api.BlobDesc): Input `Blob`.
        mean (oneflow_api.BlobDesc): Mean over neurons.
        inv_variance (oneflow_api.BlobDesc): Variance over neurons.
        begin_norm_axis (int, optional): An integer specifies which axis to normalize at first. Defaults to 1.
        name (Optional[str], optional): This layer's name. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: Gradient with respect to input `Blob`.
    """
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
    dy: oneflow_api.BlobDesc,
    norm: oneflow_api.BlobDesc,
    gamma: oneflow_api.BlobDesc,
    begin_params_axis: int = -1,
    name: str = "LayerNormParamGrad",
) -> Tuple[oneflow_api.BlobDesc, oneflow_api.BlobDesc, oneflow_api.BlobDesc]:
    r"""Backward pass for layer normalization

    Args:
        dy (oneflow_api.BlobDesc): Upstream derivstives.
        norm (oneflow_api.BlobDesc): Normalized output.
        gamma (oneflow_api.BlobDesc): Scale parameter.
        begin_params_axis (int, optional): From which parameters to begin with. Defaults to -1.
        name (Optional[str], optional): This layer's name. Defaults to 'LayerNormParamGrad'.

    Returns:
        Tuple[oneflow_api.BlobDesc]:
                normalized_diff: Gradient with respect to input `Blob`.
                beta_diff: Gradient with respect to shift parameter beta.
                gamma_diff: Gradient with respect to scale parameter gamma.
    """
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


def _get_batch_normalization_variables(
    name,
    gamma_name,
    beta_name,
    moving_mean_name,
    moving_variance_name,
    center,
    scale,
    params_shape,
    params_dtype,
    trainable,
    beta_initializer,
    beta_regularizer,
    gamma_initializer,
    gamma_regularizer,
    moving_mean_initializer,
    moving_variance_initializer,
):
    def get_beta_var(name):
        if center:
            beta = flow.get_variable(
                name=name,
                shape=params_shape,
                dtype=params_dtype,
                initializer=beta_initializer or flow.zeros_initializer(),
                regularizer=beta_regularizer,
                trainable=trainable,
                distribute=oneflow_api.sbp.broadcast(),
                reuse=False,
            )
        else:
            beta = flow.constant(0, dtype=params_dtype, shape=params_shape, name=name)
        return beta

    if beta_name is None:
        with flow.scope.namespace(name):
            beta = get_beta_var("beta")
    else:
        beta = get_beta_var(beta_name)

    def get_gamma_var(name):
        if scale:
            gamma = flow.get_variable(
                name=name,
                shape=params_shape,
                dtype=params_dtype,
                initializer=gamma_initializer or flow.ones_initializer(),
                regularizer=gamma_regularizer,
                trainable=trainable,
                distribute=oneflow_api.sbp.broadcast(),
                reuse=False,
            )
        else:
            gamma = flow.constant(1, dtype=params_dtype, shape=params_shape, name=name)
        return gamma

    if gamma_name is None:
        with flow.scope.namespace(name):
            gamma = get_gamma_var("gamma")
    else:
        gamma = get_gamma_var(gamma_name)

    def get_moving_mean_var(name):
        moving_mean = flow.get_variable(
            name=name,
            shape=params_shape,
            dtype=params_dtype,
            initializer=moving_mean_initializer or flow.zeros_initializer(),
            trainable=False,
            distribute=oneflow_api.sbp.broadcast(),
            reuse=False,
        )
        return moving_mean

    if moving_mean_name is None:
        with flow.scope.namespace(name):
            moving_mean = get_moving_mean_var("moving_mean")
    else:
        moving_mean = get_moving_mean_var(moving_mean_name)

    def get_moving_variance_var(name):
        moving_variance = flow.get_variable(
            name=name,
            shape=params_shape,
            dtype=params_dtype,
            initializer=moving_variance_initializer or flow.ones_initializer(),
            trainable=False,
            distribute=oneflow_api.sbp.broadcast(),
            reuse=False,
        )
        return moving_variance

    if moving_variance_name is None:
        with flow.scope.namespace(name):
            moving_variance = get_moving_variance_var("moving_variance")
    else:
        moving_variance = get_moving_variance_var(moving_variance_name)

    return beta, gamma, moving_mean, moving_variance


@oneflow_export("layers.batch_normalization")
def batch_normalization(
    inputs: oneflow_api.BlobDesc,
    axis: int = -1,
    momentum: float = 0.99,
    epsilon: float = 0.001,
    center: bool = True,
    scale: bool = True,
    beta_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    gamma_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    beta_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    gamma_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    moving_mean_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    moving_variance_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    trainable: bool = True,
    training: bool = True,
    name: str = "BatchNorm",
    gamma_name: Optional[str] = None,
    beta_name: Optional[str] = None,
    moving_mean_name: Optional[str] = None,
    moving_variance_name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""The BatchNormalization Layer. 

    This layer can be used in conv or dense layer.

    The input data will be normalized by the mean and variance of the current batch data

    Args:
        inputs (oneflow_api.BlobDesc): Input `Blob`.
        axis (int, optional): An int specifies the axis that should be normalized . Default is -1, which normalizes the last axis.
        momentum (float, optional):  A float specifies the momentum for the moving average. Defaults to 0.99.
        epsilon (float, optional): A small float added to avoid division by zero. Defaults to 0.001.
        center (bool, optional): A boolean specifies whether to add offset to normalized `Blob`. Defaults to True.
        scale (bool, optional): A boolean specifies whether to multiply normalized `Blob` by gamma. Defaults to True.
        beta_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for beta. Defaults to None.
        gamma_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for gamma. Defaults to None.
        beta_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for beta. Defaults to None.
        gamma_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for gamma. Defaults to None.
        moving_mean_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for moving mean. Defaults to None.
        moving_variance_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for moving variance. Defaults to None.
        trainable (bool, optional): A boolean specifies whether to train variables. Defaults to True.
        training (bool, optional): A boolean specifies whether now is training the model. Defaults to True.
        name (Optional[str], optional): This layer's name. Defaults to None.
        gamma_name (Optional[str], optional): This gamma's name. Defaults to None.
        beta_name (Optional[str], optional): This beta's name. Defaults to None.
        moving_mean_name (Optional[str], optional): This moving_mean's name. Defaults to None.
        moving_variance_name (Optional[str], optional): This moving_var's name. Defaults to None.

    Returns:
        oneflow_api.BlobDesc:  A `Blob` with same shape of input.

    Raises:
        ValueError: If axis is out of dimension of input.

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def batch_norm_Job(x: tp.Numpy.Placeholder((1, 64, 128, 128))
        ) -> tp.Numpy:
            initializer = flow.truncated_normal(0.1)
            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=2,
                padding='SAME',
                kernel_initializer=initializer,
                name="Conv2d"
            )
            batch_norm = flow.layers.batch_normalization(
                conv2d,
                axis=1
            )
            return batch_norm


        x = np.random.randn(1, 64, 128, 128).astype(np.float32)
        out = batch_norm_Job(x)

        # out.shape (1, 128, 64, 64)

    """
    if axis < 0:
        axis += len(inputs.shape)
    assert axis >= 0 and axis < len(inputs.shape)

    params_shape = [inputs.shape[axis]]
    # Float32 required to avoid precision-loss when using fp16 input/output
    params_dtype = flow.float32 if inputs.dtype == flow.float16 else inputs.dtype

    if not flow.current_global_function_desc().IsTrainable() or not trainable:
        training = False

    beta, gamma, moving_mean, moving_variance = _get_batch_normalization_variables(
        name,
        gamma_name,
        beta_name,
        moving_mean_name,
        moving_variance_name,
        center,
        scale,
        params_shape,
        params_dtype,
        trainable,
        beta_initializer,
        beta_regularizer,
        gamma_initializer,
        gamma_regularizer,
        moving_mean_initializer,
        moving_variance_initializer,
    )

    if flow.current_scope().device_parallel_desc_symbol.device_tag == "cpu":
        if training:
            reduce_axis = []
            for dim in range(len(inputs.shape)):
                if dim != axis:
                    reduce_axis.append(dim)
            mean, variance = flow.nn.moments(inputs, reduce_axis, keepdims=False)

            def update_moving(moving, this_batch):
                moving_identity = flow.identity(moving)
                flow.assign(
                    moving, momentum * moving_identity + (1 - momentum) * this_batch
                )

            update_moving(moving_mean, mean)
            update_moving(moving_variance, variance)

            return flow.nn.batch_normalization(
                x=inputs,
                mean=mean,
                variance=variance,
                offset=beta,
                scale=gamma,
                variance_epsilon=epsilon,
                axis=axis,
                name=name,
            )
        else:
            mean = moving_mean
            variance = moving_variance
            return flow.nn.batch_normalization(
                x=inputs,
                mean=mean,
                variance=variance,
                offset=beta,
                scale=gamma,
                variance_epsilon=epsilon,
                axis=axis,
                name=name,
            )
    else:
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


@oneflow_export("layers.batch_normalization_add_relu")
def batch_normalization_add_relu(
    inputs: oneflow_api.BlobDesc,
    addend: Optional[oneflow_api.BlobDesc] = None,
    axis: int = -1,
    momentum: float = 0.99,
    epsilon: float = 0.001,
    center: bool = True,
    scale: bool = True,
    beta_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    gamma_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    beta_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    gamma_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    moving_mean_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    moving_variance_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    trainable: bool = True,
    training: bool = True,
    name: str = "BatchNorm",
    gamma_name: Optional[str] = None,
    beta_name: Optional[str] = None,
    moving_mean_name: Optional[str] = None,
    moving_variance_name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Fused flow.layers.batch_normalization + flow.math.add + flow.math.relu

    Args:
        inputs (oneflow_api.BlobDesc): Input `Blob`.
        addend (oneflow_api.BlobDesc): `Blob` add to batch_normalization output.
        axis (int, optional): An int specifies the axis that should be normalized . Default is -1, which normalizes the last axis.
        momentum (float, optional):  A float specifies the momentum for the moving average. Defaults to 0.99.
        epsilon (float, optional): A small float added to avoid division by zero. Defaults to 0.001.
        center (bool, optional): A boolean specifies whether to add offset to normalized `Blob`. Defaults to True.
        scale (bool, optional): A boolean specifies whether to multiply normalized `Blob` by gamma. Defaults to True.
        beta_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for beta. Defaults to None.
        gamma_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for gamma. Defaults to None.
        beta_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for beta. Defaults to None.
        gamma_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for gamma. Defaults to None.
        moving_mean_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for moving mean. Defaults to None.
        moving_variance_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for moving variance. Defaults to None.
        trainable (bool, optional): A boolean specifies whether to train variables. Defaults to True.
        training (bool, optional): A boolean specifies whether now is training the model. Defaults to True.
        name (Optional[str], optional): This layer's name. Defaults to None.
        gamma_name (Optional[str], optional): This gamma's name. Defaults to None.
        beta_name (Optional[str], optional): This beta's name. Defaults to None.
        moving_mean_name (Optional[str], optional): This moving_mean's name. Defaults to None.
        moving_variance_name (Optional[str], optional): This moving_var's name. Defaults to None.

    Returns:
        oneflow_api.BlobDesc:  A `Blob` with same shape of input.

    Raises:
        ValueError: If axis is out of dimension of input.

    """
    if not flow.current_global_function_desc().IsTrainable() or not trainable:
        training = False

    if (
        not training
        or flow.current_scope().device_parallel_desc_symbol.device_tag == "cpu"
    ):
        out = flow.layers.batch_normalization(
            inputs,
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            trainable=trainable,
            training=training,
            name=name,
        )
        with flow.scope.namespace("BatchNormAddRelu"):
            if addend is not None:
                out = out + addend
            return flow.math.relu(out)

    if axis < 0:
        axis += len(inputs.shape)
    assert 0 <= axis < len(inputs.shape)

    params_shape = [inputs.shape[axis]]
    # Float32 required to avoid precision-loss when using fp16 input/output
    params_dtype = flow.float32 if inputs.dtype == flow.float16 else inputs.dtype

    beta, gamma, moving_mean, moving_variance = _get_batch_normalization_variables(
        name,
        gamma_name,
        beta_name,
        moving_mean_name,
        moving_variance_name,
        center,
        scale,
        params_shape,
        params_dtype,
        trainable,
        beta_initializer,
        beta_regularizer,
        gamma_initializer,
        gamma_regularizer,
        moving_mean_initializer,
        moving_variance_initializer,
    )

    builder = (
        flow.user_op_builder(name)
        .Op("normalization_add_relu")
        .Input("x", [inputs])
        .Input("moving_mean", [moving_mean])
        .Input("moving_variance", [moving_variance])
        .Input("gamma", [gamma])
        .Input("beta", [beta])
        .Output("y")
        .Output("mean")
        .Output("inv_variance")
        .Output("reserve_space")
        .Attr("axis", axis)
        .Attr("epsilon", epsilon)
        .Attr("momentum", momentum)
    )
    if addend is not None:
        builder = builder.Input("addend", [addend])
    return builder.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("layers.batch_normalization_relu")
def batch_normalization_relu(
    inputs: oneflow_api.BlobDesc,
    axis: int = -1,
    momentum: float = 0.99,
    epsilon: float = 0.001,
    center: bool = True,
    scale: bool = True,
    beta_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    gamma_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    beta_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    gamma_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    moving_mean_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    moving_variance_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    trainable: bool = True,
    training: bool = True,
    name: str = "BatchNorm",
) -> oneflow_api.BlobDesc:
    r"""Fused flow.layers.batch_normalization + flow.math.relu

Args:
    inputs (oneflow_api.BlobDesc): Input `Blob`.
    axis (int, optional): An int specifies the axis that should be normalized . Default is -1, which normalizes the last axis.
    momentum (float, optional):  A float specifies the momentum for the moving average. Defaults to 0.99.
    epsilon (float, optional): A small float added to avoid division by zero. Defaults to 0.001.
    center (bool, optional): A boolean specifies whether to add offset to normalized `Blob`. Defaults to True.
    scale (bool, optional): A boolean specifies whether to multiply normalized `Blob` by gamma. Defaults to True.
    beta_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for beta. Defaults to None.
    gamma_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for gamma. Defaults to None.
    beta_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for beta. Defaults to None.
    gamma_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): Regularizer for gamma. Defaults to None.
    moving_mean_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for moving mean. Defaults to None.
    moving_variance_initializer (Optional[initializer_conf_util.InitializerConf], optional): Initializer for moving variance. Defaults to None.
    trainable (bool, optional): A boolean specifies whether to train variables. Defaults to True.
    training (bool, optional): A boolean specifies whether now is training the model. Defaults to True.
    name (Optional[str], optional): This layer's name. Defaults to None.

Returns:
    oneflow_api.BlobDesc:  A `Blob` with same shape of input.

Raises:
    ValueError: If axis is out of dimension of input.

"""
    return flow.layers.batch_normalization_add_relu(
        inputs,
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        trainable=trainable,
        training=training,
        name=name,
    )


@oneflow_export("layers.upsample_2d")
def upsample(
    x: oneflow_api.BlobDesc,
    size: Sequence[int] = (2, 2),
    data_format: str = "NCHW",
    interpolation: str = "nearest",
    name: str = "Upsample2D",
):
    r"""The Upsample Layer, this layer can upsample the feature map to a specified scale. 

    Args:
        x ([type]): Input `Blob`.
        size (tuple, optional): (height_scale, width_scale)  Defaults to (2, 2).
        data_format (str, optional): A string specifies the format of the input `Blob`, one of "NCHW" or "NHWC" (default: "NCHW"). "NCHW" cooresponds to channels_first, i.e. the input `Blob` with shape (batch_size, channels, height, width).
                        "NHWC" cooresponds to channels_last, i.e. the input `Blob` with shape (batch_size, height, width, channels).. Defaults to "NCHW".
        interpolation (str, optional): Image interpolation algorithm to enlarge the image size. Defaults to "nearest". "nearest" and "bilinear" are available now.
        name ([type], optional): This layer's name. Defaults to None.

    Raises:
        ValueError: interpolation must be "nearest" or "bilinear".
        ValueError: data_format must be "NHWC" or "NCHW"

    Returns:
        [type]: oneflow_api.BlobDesc:  A `Blob` which is the upsampled `x`. If `size` is (2, 2), the shape of return value is [N, C, 2H, 2W].

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def upsample_Job(x: tp.Numpy.Placeholder((1, 32, 32, 32))
        ) -> tp.Numpy:
            upsample = flow.layers.upsample_2d(
                x,
                size=(2, 2),
                name="Upsample1"
            )
            return upsample


        x = np.random.randn(1, 32, 32, 32).astype(np.float32)
        out = upsample_Job(x)

        # out.shape (1, 32, 64, 64)

    """
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
