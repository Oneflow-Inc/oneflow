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
import collections
import os
import random
import sys
from typing import List, Optional, Sequence, Tuple, Union

import oneflow as flow
import oneflow._oneflow_internal
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.framework.distribute as distribute_util
import oneflow.framework.id_util as id_util
import oneflow.framework.interpret_util as interpret_util
import oneflow.framework.module as module_util
import oneflow.framework.remote_blob as remote_blob_util

IntPair = Tuple[int, int]


def calc_same_padding(input_size, filter_size, dilation_rate, stride):
    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    output_size = (input_size + stride - 1) // stride
    padding_needed = max(
        0, int((output_size - 1) * stride + effective_filter_size - input_size)
    )
    return padding_needed


def get_dhw_offset(channel_pos):
    if channel_pos == "channels_first":
        return 2
    else:
        return 1


def check_conv_cudnn_padding_support(
    input_size, pad, filter_size, dilation_rate, stride, is_dynamic
):
    assert len(pad) == 2
    if pad[0] == pad[1]:
        return True
    elif is_dynamic or pad[0] < pad[1] or pad[0] - pad[1] > 1:
        return False
    else:
        effective_filter_size = (filter_size - 1) * dilation_rate + 1
        cudnn_output_size = (
            input_size + 2 * pad[0] - effective_filter_size + stride
        ) // stride
        output_size = (
            input_size + pad[0] + pad[1] - effective_filter_size + stride
        ) // stride
        return cudnn_output_size == output_size


def check_ndim_conv_cudnn_padding_support(
    inputs_shape,
    ndim_pads_list,
    kernel_sizes,
    dilations,
    strides,
    dhw_offset,
    is_dynamic,
):
    ndims = len(ndim_pads_list)
    for i in range(ndims):
        cudnn_support = check_conv_cudnn_padding_support(
            inputs_shape[dhw_offset + i],
            ndim_pads_list[i],
            kernel_sizes[i],
            dilations[i],
            strides[i],
            is_dynamic,
        )
        if not cudnn_support:
            return False
    return True


def get_ndim_pads_list(padding, dhw_offset, ndims):
    pads_list = []
    for i in range(len(padding)):
        pad = padding[i]
        if isinstance(pad, int):
            pad = [pad, pad]
        elif isinstance(pad, (list, tuple)):
            assert len(pad) == 2
            pad = [pad[0], pad[1]]
        else:
            raise ValueError("padding must be list tuple or int")
        if i in range(dhw_offset, dhw_offset + ndims):
            pads_list.append(pad)
        else:
            assert pad == [0, 0]
    return pads_list


def calc_ndim_same_padding(
    input_shape, padding, kernel_sizes, dilations, strides, dhw_offset
):
    ndim_padding_needed = []
    ndims = len(kernel_sizes)
    for i in range(ndims):
        ndim_padding_needed.append(
            calc_same_padding(
                input_shape[dhw_offset + i], kernel_sizes[i], dilations[i], strides[i]
            )
        )
    pads_small = [padding_needed // 2 for padding_needed in ndim_padding_needed]
    pads_large = [ndim_padding_needed[i] - pads_small[i] for i in range(ndims)]
    if padding.upper() == "SAME_LOWER":
        return [[pads_large[i], pads_small[i]] for i in range(ndims)]
    elif padding.upper() == "SAME_UPPER":
        return [[pads_small[i], pads_large[i]] for i in range(ndims)]
    else:
        raise NotImplementedError


def calc_conv_padding(inputs, padding, data_format, kernel_sizes, dilations, strides):
    ndims = len(inputs.shape) - 2
    assert len(kernel_sizes) == ndims
    assert len(dilations) == ndims
    assert len(strides) == ndims
    is_dynamic = inputs.is_dynamic
    channel_pos = "channels_first" if data_format.startswith("NC") else "channels_last"
    dhw_offset = get_dhw_offset(channel_pos)
    ndim_pads_list = []
    if isinstance(padding, str):
        padding = "SAME_LOWER" if padding.upper() == "SAME" else padding
        assert padding.upper() in ["VALID", "SAME_LOWER", "SAME_UPPER"]
        if padding.upper() == "VALID":
            return_pads_list = [[0, 0]] * ndims
            return (inputs, return_pads_list)
        elif is_dynamic:
            return_pads_list = [[0, 0]] * ndims
            inputs = flow.same_padding(
                inputs,
                padding.lower(),
                data_format=data_format,
                kernel_size=kernel_sizes,
                strides=strides,
                dilation_rate=dilations,
            )
            return (inputs, return_pads_list)
        else:
            ndim_pads_list = calc_ndim_same_padding(
                inputs.shape, padding, kernel_sizes, dilations, strides, dhw_offset
            )
            assert len(ndim_pads_list) == ndims
    elif isinstance(padding, (list, tuple)):
        assert len(padding) == ndims + 2
        ndim_pads_list = get_ndim_pads_list(padding, dhw_offset, ndims)
        assert len(ndim_pads_list) == ndims
    else:
        raise ValueError("padding must be str or a list.")
    cudnn_padding_support = check_ndim_conv_cudnn_padding_support(
        inputs.shape,
        ndim_pads_list,
        kernel_sizes,
        dilations,
        strides,
        dhw_offset,
        is_dynamic,
    )
    if cudnn_padding_support:
        return (inputs, ndim_pads_list)
    else:
        pad_op_list = [[0, 0]] * (ndims + 2)
        for i in range(ndims):
            pad_op_list[dhw_offset + i] = ndim_pads_list[i]
        inputs = flow.pad(inputs, paddings=pad_op_list)
        return_pads_list = [[0, 0]] * ndims
        return (inputs, return_pads_list)


class ConvUtil(object):
    @classmethod
    def split(cls, x, axis, split_num):
        split_len = x.shape[axis] // split_num
        result_list = []
        slice_begin = [0] * len(x.shape)
        slice_size = [-1] * len(x.shape)
        slice_size[axis] = split_len
        for i in range(split_num):
            slice_begin[axis] = i * split_len
            result = flow.slice(x, slice_begin, slice_size)
            result_list.append(result)
        return result_list


def conv_op(
    conv_type,
    inputs,
    filters,
    bias,
    padding_before,
    channel_pos,
    kernel_size_list,
    strides,
    dilations,
    groups,
    name,
):
    op_builder = (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Conv_"))
        .Op(conv_type)
        .Input("in", [inputs])
        .Input("weight", [filters])
        .Output("out")
        .Attr("filters", filters.shape[0])
        .Attr("padding_before", padding_before)
        .Attr("data_format", channel_pos)
        .Attr("kernel_size", kernel_size_list)
        .Attr("strides", strides)
        .Attr("dilation_rate", dilations)
        .Attr("groups", groups)
    )
    if bias is not None:
        op_builder = op_builder.Input("bias", [bias])
    return op_builder.Build().InferAndTryRun().RemoteBlobList()[0]


def conv1d(
    input: oneflow._oneflow_internal.BlobDesc,
    filters: oneflow._oneflow_internal.BlobDesc,
    strides: Union[int, Tuple[int]],
    padding: Union[str, Tuple[IntPair, IntPair, IntPair]],
    data_format: str = "NCW",
    dilations: Optional[Union[int, Tuple[int]]] = None,
    groups: int = 1,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """1D convolution layer.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): A 3D input `Blob`. [batch_num, channel, width]
        filters (oneflow._oneflow_internal.BlobDesc): A `Blob` with the same type as `input` and has the shape [out_channels, in_channels//groups, filter_width] for `NCW`, or [out_channels, filter_width, in_channels//groups] for `NWC`
        strides (Union[int, Tuple[int]]): An int or list of `ints` that has length `1`. The stride of the sliding window for each dimension of `input`.
        padding (Union[str, Tuple[IntPair, IntPair, IntPair]]): padding: `string` `"SAME"` or `"SAME_LOWER"` or `"SAME_UPPER"` or `"VALID" or Tuple[IntPair, IntPair, IntPair]` indicating the type of padding algorithm to use, or a list indicating the explicit paddings at the start and end of each dimension.
        data_format (str, optional): `"NWC" or "NCW"`. Defaults to `"NCW"`.
        dilations (Optional[Union[int, Tuple[int]]], optional): An int or list of `ints` that has length `1`. The dilation factor for each dimension of `input`. Defaults to None.
        groups (int, optional): int value greater than 0. Defaults to 1.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Raises:
        ValueError: strides must be an int or a list.
        ValueError: padding must be "SAME" or "SAME_LOWER" or "SAME_UPPER" or "VALID" or Tuple[IntPair, IntPair, IntPair, IntPair].
        ValueError: data_format must be "NWC" or "NCW".
        ValueError: dilations must be an int or a list.
        ValueError: invalid data_format.
        ValueError: data_format NWC not support groups > 1
        ValueError: invalid data_format.

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` with the same type as `input` and the same outer batch shape.

    Note:

        This api is more flexible, if you're new to OneFlow, it's more recommend to use `oneflow.layers.conv1d`

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        def conv1d(input, filters, kernel_size, strides, padding, name):
            input_shape = input.shape
            weight_initializer = flow.truncated_normal(0.1)
            weight_regularizer = flow.regularizers.l2(0.0005)
            weight_shape = (filters,
                            input_shape[1],
                            kernel_size)

            weight = flow.get_variable(
                name + "-weight",
                shape=weight_shape,
                initializer=weight_initializer,
                regularizer=weight_regularizer,
            )
            return flow.nn.conv1d(input, weight, strides, padding, name=name)


        @flow.global_function()
        def conv1d_Job(x: tp.Numpy.Placeholder((1, 64, 32))
        ) -> tp.Numpy:
            conv = conv1d(x,
                        filters=32,
                        kernel_size=3,
                        strides=1,
                        padding='SAME',
                        name="Convlayer")
            return conv


        x = np.random.randn(1, 64, 32).astype(np.float32)
        out = conv1d_Job(x)

        # out.shape (1, 32, 32)

    """
    assert len(input.shape) == 3
    assert len(filters.shape) == 3
    if isinstance(strides, (list, tuple)):
        assert len(strides) == 1, ValueError(
            "strides length must be 1 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides]
    else:
        raise ValueError("strides must be an int or a list.")
    if data_format.upper() != "NCW" and data_format.upper() != "NWC":
        raise ValueError('data_format must be "NCW" or "NWC".')
    channel_pos = "channels_first" if data_format == "NCW" else "channels_last"
    if dilations is None:
        dilations = [1]
    elif isinstance(dilations, (list, tuple)):
        assert len(dilations) == 1, ValueError(
            "dilations length must be 1 when passed as a list."
        )
    elif isinstance(dilations, int):
        dilations = [dilations]
    else:
        raise ValueError("dilations must be an int or a list.")
    if channel_pos == "channels_first":
        kernel_size_list = filters.shape[2:3]
        in_channel_axis = 1
        filter_out_axis = 0
        filter_in_axis = 1
    elif channel_pos == "channels_last":
        kernel_size_list = filters.shape[-2:-1]
        in_channel_axis = 2
        filter_out_axis = 0
        filter_in_axis = 2
        if groups > 1:
            raise ValueError("data_format NWC not support groups > 1")
    else:
        raise ValueError("invalid data_format")
    assert isinstance(kernel_size_list, tuple)
    assert isinstance(groups, int)
    assert groups > 0
    assert groups <= filters.shape[filter_out_axis]
    assert filters.shape[filter_out_axis] % groups == 0
    assert groups <= input.shape[in_channel_axis]
    assert input.shape[in_channel_axis] % groups == 0
    assert filters.shape[filter_in_axis] == input.shape[in_channel_axis] // groups
    (inputs, pads_list) = calc_conv_padding(
        input, padding, data_format.upper(), kernel_size_list, dilations, strides
    )
    assert len(pads_list) == len(inputs.shape) - 2
    padding_before = [pad[0] for pad in pads_list]
    if (
        groups > 1
        and flow.current_scope().device_parallel_desc_symbol.device_tag == "cpu"
    ):
        in_split_list = ConvUtil.split(inputs, axis=in_channel_axis, split_num=groups)
        filter_split_list = ConvUtil.split(
            filters, axis=filter_out_axis, split_num=groups
        )
        out_list = []
        name = name if name is not None else id_util.UniqueStr("Conv1d_")
        for i in range(len(in_split_list)):
            out_list.append(
                conv_op(
                    "conv1d",
                    in_split_list[i],
                    filter_split_list[i],
                    None,
                    padding_before,
                    channel_pos,
                    kernel_size_list,
                    strides,
                    dilations,
                    groups=1,
                    name=name + str(i),
                )
            )
        return flow.concat(out_list, axis=in_channel_axis)
    else:
        return conv_op(
            "conv1d",
            inputs,
            filters,
            None,
            padding_before,
            channel_pos,
            kernel_size_list,
            strides,
            dilations,
            groups,
            name,
        )


def conv2d(
    input: oneflow._oneflow_internal.BlobDesc,
    filters: oneflow._oneflow_internal.BlobDesc,
    strides: Union[int, IntPair],
    padding: Union[str, Tuple[IntPair, IntPair, IntPair, IntPair]],
    bias: Optional[oneflow._oneflow_internal.BlobDesc] = None,
    data_format: str = "NCHW",
    dilations: Optional[Union[int, IntPair]] = None,
    groups: int = 1,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """2D convolution layer.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): A 4D input `Blob`. [batch_num, channel, height, width]
        filters (oneflow._oneflow_internal.BlobDesc): A `Blob` with the same type as `input` and has the shape `[out_channels, in_channels//groups, filter_height, filter_width] for NCHW, or [out_channels, filter_height, filter_width, in_channels//groups] for NHWC`
        strides (Union[int, IntPair]): An int or list of `ints` that has length `2`. The stride of the sliding window for each dimension of `input`.
        padding (Union[str, Tuple[IntPair, IntPair, IntPair, IntPair]]): padding: `string` `"SAME"` or `"SAME_LOWER"` or `"SAME_UPPER"` or `"VALID" or Tuple[IntPair, IntPair, IntPair, IntPair]` indicating the type of padding algorithm to use, or a list indicating the explicit paddings at the start and end of each dimension.
        data_format (str, optional): `"NHWC"` or `"NCHW"`. Defaults to `"NCHW"`.
        dilations (Optional[Union[int, IntPair]], optional): An int or list of `ints` that has length `2`. The dilation factor for each dimension of `input`. Defaults to None.
        groups (int, optional): int value greater than 0. Defaults to 1.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Raises:
        ValueError: strides must be an int or a list.
        ValueError: padding must be `"SAME"` or `"SAME_LOWER" or `"SAME_UPPER"` or `"VALID"` or Tuple[IntPair, IntPair, IntPair, IntPair].
        ValueError: data_format must be `"NHWC"` or `"NCHW"`.
        ValueError: dilations must be an int or a list.
        ValueError: invalid data_format.
        ValueError: data_format NHWC not support groups > 1
        ValueError: invalid data_format.

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` with the same type as `input` and the same outer batch shape.

    Note:

        This api is more flexible, if you're new to OneFlow, it's more recommend to use `oneflow.layers.conv2d`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        def conv2d(input, filters, kernel_size, strides, padding, name):
            input_shape = input.shape
            weight_initializer = flow.truncated_normal(0.1)
            weight_regularizer = flow.regularizers.l2(0.0005)
            weight_shape = (filters,
                            input_shape[1],
                            kernel_size[0],
                            kernel_size[1])

            weight = flow.get_variable(
                name + "-weight",
                shape=weight_shape,
                initializer=weight_initializer,
                regularizer=weight_regularizer,
            )
            return flow.nn.conv2d(input, weight, strides, padding, name=name)


        @flow.global_function()
        def conv2d_Job(x: tp.Numpy.Placeholder((1, 64, 32, 32))
        ) -> tp.Numpy:
            conv = conv2d(x,
                        filters=128,
                        kernel_size=[3, 3],
                        strides=2,
                        padding='SAME',
                        name="Convlayer")
            return conv


        x = np.random.randn(1, 64, 32, 32).astype(np.float32)
        out = conv2d_Job(x)

        # out.shape (1, 128, 16, 16)

    """
    assert len(input.shape) == 4
    assert len(filters.shape) == 4
    if bias is not None:
        assert len(bias.shape) == 1
    if isinstance(strides, (list, tuple)):
        assert len(strides) == 2, ValueError(
            "strides length must be 2 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")
    if data_format.upper() != "NCHW" and data_format.upper() != "NHWC":
        raise ValueError('data_format must be "NHWC" or "NCHW".')
    channel_pos = "channels_first" if data_format == "NCHW" else "channels_last"
    if dilations is None:
        dilations = [1, 1]
    elif isinstance(dilations, (list, tuple)):
        assert len(dilations) == 2, ValueError(
            "dilations length must be 2 when passed as a list."
        )
    elif isinstance(dilations, int):
        dilations = [dilations, dilations]
    else:
        raise ValueError("dilations must be an int or a list.")
    assert isinstance(groups, int)
    assert groups > 0
    if data_format.upper() == "NCHW":
        kernel_size_list = filters.shape[2:4]
        in_channel_axis = 1
        filter_out_axis = 0
        filter_in_axis = 1
    elif data_format.upper() == "NHWC":
        kernel_size_list = filters.shape[-3:-1]
        in_channel_axis = 3
        filter_out_axis = 0
        filter_in_axis = 3
        if (
            groups > 1
            and flow.current_scope().device_parallel_desc_symbol.device_tag == "gpu"
        ):
            raise ValueError("gpu data_format NHWC not support groups > 1")
    else:
        raise ValueError('data_format must be "NHWC" or "NCHW".')
    assert isinstance(kernel_size_list, tuple)
    (inputs, pads_list) = calc_conv_padding(
        input, padding, data_format.upper(), kernel_size_list, dilations, strides
    )
    assert len(pads_list) == len(inputs.shape) - 2
    padding_before = [pad[0] for pad in pads_list]
    assert groups <= filters.shape[filter_out_axis]
    assert filters.shape[filter_out_axis] % groups == 0
    assert groups <= inputs.shape[in_channel_axis]
    assert inputs.shape[in_channel_axis] % groups == 0
    assert filters.shape[filter_in_axis] == inputs.shape[in_channel_axis] // groups
    if bias is not None:
        assert bias.shape[filter_out_axis] == filters.shape[filter_out_axis]
    if (
        groups > 1
        and flow.current_scope().device_parallel_desc_symbol.device_tag == "cpu"
    ):
        in_split_list = ConvUtil.split(inputs, axis=in_channel_axis, split_num=groups)
        filter_split_list = ConvUtil.split(
            filters, axis=filter_out_axis, split_num=groups
        )
        bias_spilt_list = (
            ConvUtil.split(bias, axis=filter_out_axis, split_num=groups)
            if bias is not None
            else [None for _ in range(groups)]
        )
        out_list = []
        name = name if name is not None else id_util.UniqueStr("Conv2d_")
        for i in range(len(in_split_list)):
            out_list.append(
                conv_op(
                    "conv2d",
                    in_split_list[i],
                    filter_split_list[i],
                    bias_spilt_list[i],
                    padding_before,
                    channel_pos,
                    kernel_size_list,
                    strides,
                    dilations,
                    groups=1,
                    name=name + str(i),
                )
            )
        return flow.concat(out_list, axis=in_channel_axis)
    else:
        return conv_op(
            "conv2d",
            inputs,
            filters,
            bias,
            padding_before,
            channel_pos,
            kernel_size_list,
            strides,
            dilations,
            groups,
            name,
        )


def conv3d(
    input: oneflow._oneflow_internal.BlobDesc,
    filters: oneflow._oneflow_internal.BlobDesc,
    strides: Union[int, Sequence[int]],
    padding: Union[str, Tuple[IntPair, IntPair, IntPair, IntPair, IntPair]],
    data_format: str = "NCDHW",
    dilations: Optional[Union[int, Sequence[int]]] = None,
    groups: int = 1,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """3D convolution layer.

    Args:
        input (oneflow._oneflow_internal.BlobDesc):  A 5D input `Blob`. [batch_num, channel, depth, height, width]
        filters (oneflow._oneflow_internal.BlobDesc): A `Blob` with the same type as `input` and has the shape `[out_channels, in_channels//groups, filter_depth, filter_height, filter_width] for NCDHW, or [out_channels, filter_depth, filter_height, filter_width, in_channels//groups] for NDHWC`
        strides (Union[int, Sequence[int]]): An `int` or `list of ints` that has length `3`. The stride of the sliding window for each dimension of `input`.
        padding (Union[str, Tuple[IntPair, IntPair, IntPair, IntPair, IntPair]]): padding: `string` `"SAME"` or `"SAME_LOWER"` or `"SAME_UPPER"` or `"VALID"` or Tuple[IntPair, IntPair, IntPair, IntPair, IntPair]` indicating the type of padding algorithm to use, or a list indicating the explicit paddings at the start and end of each dimension.
        data_format (str, optional): `"NDHWC" or "NCDHW"`. Defaults to `"NCDHW"`.
        dilations (Optional[Union[int, Sequence[int]]], optional): An int or list of `ints` that has length `3`. The dilation factor for each dimension of `input`. Defaults to None.
        groups (int, optional): int value greater than 0. Defaults to 1.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Raises:
        ValueError: strides must be an int or a list.
        ValueError: padding must be "SAME" or "SAME_LOWER" or "SAME_UPPER" or "VALID" or Tuple[IntPair, IntPair, IntPair, IntPair, IntPair].
        ValueError: data_format must be "NDHWC" or "NCDHW".
        ValueError: dilations must be an int or a list.
        ValueError: invalid data_format.
        ValueError: data_format NDHWC not support groups > 1
        ValueError: invalid data_format.

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` with the same type as `input` and the same outer batch shape.

    Note:

        This api is more flexible, if you're new to OneFlow, it's more recommend to use `oneflow.layers.conv3d`

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        def conv3d(input, filters, kernel_size, strides, padding, name):
            input_shape = input.shape
            weight_initializer = flow.truncated_normal(0.1)
            weight_regularizer = flow.regularizers.l2(0.0005)
            weight_shape = (filters,
                            input_shape[1],
                            kernel_size[0],
                            kernel_size[1],
                            kernel_size[2])

            weight = flow.get_variable(
                name + "-weight",
                shape=weight_shape,
                initializer=weight_initializer,
                regularizer=weight_regularizer,
            )
            return flow.nn.conv3d(input, weight, strides, padding, name=name)


        @flow.global_function()
        def conv3d_Job(x: tp.Numpy.Placeholder((1, 64, 10, 16, 16))
        ) -> tp.Numpy:
            conv = conv3d(x,
                        filters=128,
                        kernel_size=[3, 3, 3],
                        strides=1,
                        padding='SAME',
                        name="Convlayer")
            return conv


        x = np.random.randn(1, 64, 10, 16, 16).astype(np.float32)
        out = conv3d_Job(x)

        # out.shape (1, 128, 10, 16, 16)

    """
    need_transpose = 0
    if data_format.upper() == "NDHWC":
        need_transpose = 1
        data_format = "NCDHW"
    if need_transpose:
        input = flow.transpose(input, perm=[0, 4, 1, 2, 3])
        filters = flow.transpose(filters, perm=[0, 4, 1, 2, 3])
        if isinstance(padding, (list, tuple)):
            padding = list(padding)
            (padding[1], padding[4]) = (padding[4], padding[1])
    assert len(input.shape) == 5
    assert len(filters.shape) == 5
    if isinstance(strides, (list, tuple)):
        assert len(strides) == 3, ValueError(
            "strides length must be 3 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")
    if data_format.upper() != "NCDHW" and data_format.upper() != "NDHWC":
        raise ValueError('data_format must be "NDHWC" or "NCDHW".')
    channel_pos = "channels_first" if data_format == "NCDHW" else "channels_last"
    if dilations is None:
        dilations = [1, 1, 1]
    elif isinstance(dilations, (list, tuple)):
        assert len(dilations) == 3, ValueError(
            "dilations length must be 3 when passed as a list."
        )
    elif isinstance(dilations, int):
        dilations = [dilations, dilations, dilations]
    else:
        raise ValueError("dilations must be an int or a list.")
    if channel_pos == "channels_first":
        kernel_size_list = filters.shape[2:5]
        in_channel_axis = 1
        filter_out_axis = 0
        filter_in_axis = 1
    elif channel_pos == "channels_last":
        kernel_size_list = filters.shape[-4:-1]
        in_channel_axis = 4
        filter_out_axis = 0
        filter_in_axis = 4
        if groups > 1:
            raise ValueError("data_format NDHWC not support groups > 1")
    else:
        raise ValueError("invalid data_format")
    assert isinstance(kernel_size_list, tuple)
    assert isinstance(groups, int)
    assert groups > 0
    assert groups <= filters.shape[filter_out_axis]
    assert filters.shape[filter_out_axis] % groups == 0
    assert groups <= input.shape[in_channel_axis]
    assert input.shape[in_channel_axis] % groups == 0
    assert filters.shape[filter_in_axis] == input.shape[1] // groups
    (inputs, pads_list) = calc_conv_padding(
        input, padding, data_format.upper(), kernel_size_list, dilations, strides
    )
    assert len(pads_list) == len(inputs.shape) - 2
    padding_before = [pad[0] for pad in pads_list]
    if (
        groups > 1
        and flow.current_scope().device_parallel_desc_symbol.device_tag == "cpu"
    ):
        in_split_list = ConvUtil.split(inputs, axis=in_channel_axis, split_num=groups)
        filter_split_list = ConvUtil.split(
            filters, axis=filter_out_axis, split_num=groups
        )
        out_list = []
        name = name if name is not None else id_util.UniqueStr("Conv3d_")
        for i in range(len(in_split_list)):
            out_list.append(
                conv_op(
                    "conv3d",
                    in_split_list[i],
                    filter_split_list[i],
                    None,
                    padding_before,
                    channel_pos,
                    kernel_size_list,
                    strides,
                    dilations,
                    groups=1,
                    name=name + str(i),
                )
            )
        output = flow.concat(out_list, axis=in_channel_axis)
    else:
        output = conv_op(
            "conv3d",
            inputs,
            filters,
            None,
            padding_before,
            channel_pos,
            kernel_size_list,
            strides,
            dilations,
            groups,
            name,
        )
    if need_transpose:
        output = flow.transpose(output, perm=[0, 2, 3, 4, 1])
    return output


def moments(
    x: oneflow._oneflow_internal.BlobDesc,
    axes: List[int],
    keepdims: Optional[bool] = False,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator computes the mean and variance value of input Blob.

    Args:
        x (oneflow._oneflow_internal.BlobDesc): A Blob
        axes (List): Array of ints. Axes along which to compute the mean and variance
        keepdims (bool, optional): Whether to keep the same dimensanality as the input x. Defaults to False.
        name (str, optional): The operator's name. Defaults to None.

    Returns:
        remote_blob: Two Blobs, mean and variance.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp
        from typing import Tuple


        @flow.global_function()
        def moments_Job(x: tp.Numpy.Placeholder((5,))
        ) -> Tuple[tp.Numpy, tp.Numpy]:
            return flow.nn.moments(x, axes=[0])


        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        mean, variance = moments_Job(x)

        # mean: [3.]
        # variance: [2.]

    """
    assert isinstance(axes, list)
    if name is None:
        name = id_util.UniqueStr("Moments_")
    with flow.scope.namespace(name):
        return (
            flow.math.reduce_mean(x, axis=axes, keepdims=keepdims),
            flow.math.reduce_variance(x, axis=axes, keepdims=keepdims),
        )


def instance_normalization1d(
    x: oneflow._oneflow_internal.BlobDesc,
    eps: float = 1e-05,
    affine: bool = True,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Applies Instance Normalization over a 3D input.

    Args:
        x (oneflow._oneflow_internal.BlobDesc): 3D input tensor with NCL data layout.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        affine (bool): A boolean value that when set to True, this module has learnable affine parameters,
                       initialized the same way as done for batch normalization. Default: True.
        name (Optional[str], optional): Name of this op.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The normalized input tensor.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def instance_norm_Job(x: tp.Numpy.Placeholder((4, 2, 32))
        ) -> tp.Numpy:
            instance_norm = flow.nn.InstanceNorm1d(
                x,
                eps=1e-5,
                affine=True,
            )
            return instance_norm

        x = np.random.random(size=(4, 2, 32)).astype(np.float32)
        out = instance_norm_Job(x)

    """
    assert len(x.shape) == 3
    if name is None:
        name = id_util.UniqueStr("InstanceNorm1D_")
    channel = x.shape[1]
    (mean, variance) = flow.nn.moments(x, [2], keepdims=True)
    normalized = (x - mean) / flow.math.sqrt(variance + eps)
    if affine == True:
        gamma = flow.get_variable(
            name + "_gamma",
            shape=(1, channel, 1),
            dtype=x.dtype,
            initializer=flow.ones_initializer(),
            trainable=True,
        )
        beta = flow.get_variable(
            name + "_beta",
            shape=(1, channel, 1),
            dtype=x.dtype,
            initializer=flow.zeros_initializer(),
            trainable=True,
        )
        return gamma * normalized + beta
    else:
        return normalized


def instance_normalization2d(
    x: oneflow._oneflow_internal.BlobDesc,
    eps: float = 1e-05,
    affine: bool = True,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Applies Instance Normalization over a 4D input.

    Args:
        x (oneflow._oneflow_internal.BlobDesc): 4D input tensor with NCHW data layout.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        affine (bool): A boolean value that when set to True, this module has learnable affine parameters,
                       initialized the same way as done for batch normalization. Default: True.
        name (Optional[str], optional): Name of this op.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The normalized input tensor.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def instance_norm_Job(x: tp.Numpy.Placeholder((4, 2, 32, 32))
        ) -> tp.Numpy:
            instance_norm = flow.nn.InstanceNorm2d(
                x,
                eps=1e-5,
                affine=True,
            )
            return instance_norm

        x = np.random.random(size=(4, 2, 32, 32)).astype(np.float32)
        out = instance_norm_Job(x)

    """
    assert len(x.shape) == 4
    if name is None:
        name = id_util.UniqueStr("InstanceNorm2D_")
    reshape_to_1d = flow.reshape(x, shape=[x.shape[0], x.shape[1], -1])
    normalized_1d_out = flow.nn.InstanceNorm1d(
        reshape_to_1d, eps=eps, affine=affine, name=name
    )
    reshape_back_to_2d = flow.reshape(normalized_1d_out, shape=list(x.shape))
    return reshape_back_to_2d


def batch_normalization(
    x: oneflow._oneflow_internal.BlobDesc,
    mean: oneflow._oneflow_internal.BlobDesc,
    variance: oneflow._oneflow_internal.BlobDesc,
    offset: Optional[oneflow._oneflow_internal.BlobDesc] = None,
    scale: Optional[oneflow._oneflow_internal.BlobDesc] = None,
    variance_epsilon: Optional[float] = 1e-05,
    axis: int = 1,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This op does not fully align with tf.nn.batch_normalization.

    The `mean`, `variable`, `offset` and `scale` are always 1D. Users need to specify `axis` to 1 for NCHW data format.

    Args:
        x (oneflow._oneflow_internal.BlobDesc): Input `Blob` of arbitrary dimensionality.
        mean (oneflow._oneflow_internal.BlobDesc): A 1D mean `Blob`.
        variance (oneflow._oneflow_internal.BlobDesc):   A 1D variance `Blob`.
        offset (Optional[oneflow._oneflow_internal.BlobDesc]): An 1D offset `Blob`, often denoted  in equations, or None. If present, will be added to the normalized `Blob`.
        scale (Optional[oneflow._oneflow_internal.BlobDesc]): A 1D scale `Blob`, often denoted  in equations, or None. If present, the scale is applied to the normalized `Blob`.
        variance_epsilon (float):   A small float number to avoid dividing by 0.
        axis (int, optional): 1 for '`NCHW'` data format. Defaults to 1.
        name (Optional[str], optional): This operator's name.

    Returns:
        oneflow._oneflow_internal.BlobDesc:  the normalized, scaled, offset `Blob`.

    Note:

        This api is more flexible, if you're new to OneFlow, it's more recommend to use `oneflow.layers.batch_normalization`

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def batch_norm_Job(x: tp.Numpy.Placeholder((1, 5))
        ) -> tp.Numpy:
            bn_mean, bn_variance = flow.nn.moments(x, axes=[1])
            batch_norm = flow.nn.batch_normalization(
                x,
                mean=bn_mean,
                variance=bn_variance,
                axis=0
            )
            return batch_norm


        x = np.array([[1, 2, 3, 4, 5]]).astype(np.float32)
        out = batch_norm_Job(x)

        # out [[-1.41421  -0.707105  0.        0.707105  1.41421 ]]

    """
    assert axis >= -len(x.shape) and axis < len(x.shape)
    if axis < 0:
        axis += len(x.shape)
    if name is None:
        name = id_util.UniqueStr("BatchNorm_")
    params_shape = [x.shape[axis]]
    if flow.current_scope().device_parallel_desc_symbol.device_tag == "cpu":
        if len(mean.shape) == 1:
            nd_params_shape = [1] * len(x.shape)
            nd_params_shape[axis] = params_shape[0]
            mean = flow.reshape(mean, nd_params_shape)
            variance = flow.reshape(variance, nd_params_shape)
            if scale:
                scale = flow.reshape(scale, nd_params_shape)
            if offset:
                offset = flow.reshape(offset, nd_params_shape)
        elif len(mean.shape) == len(x.shape):
            pass
        else:
            raise ValueError(
                "shape of mean and variance should be 1D or has number of axes and x's"
            )
        variance += variance_epsilon
        std_inv = flow.math.rsqrt(variance)
        normalized = (x - mean) * std_inv
        affined = normalized
        if scale:
            affined *= scale
        if offset:
            affined += offset
        return affined
    elif flow.current_scope().device_parallel_desc_symbol.device_tag == "gpu":
        params_dtype = flow.float32 if x.dtype == flow.float16 else x.dtype
        if scale is None:
            scale = flow.constant(
                1, dtype=params_dtype, shape=params_shape, name="gamma"
            )
        if offset is None:
            offset = flow.constant(
                0, dtype=params_dtype, shape=params_shape, name="beta"
            )
        builder = (
            flow.user_op_builder(name)
            .Op("normalization")
            .Input("x", [x])
            .Input("moving_mean", [mean])
            .Input("moving_variance", [variance])
            .Input("gamma", [scale])
            .Input("beta", [offset])
            .Output("y")
            .Attr("axis", axis)
            .Attr("epsilon", variance_epsilon)
            .Attr("training", False)
            .Attr("momentum", 0.0)
        )
        return builder.Build().InferAndTryRun().RemoteBlobList()[0]
    else:
        raise NotImplementedError


def layer_norm(
    inputs: oneflow._oneflow_internal.BlobDesc,
    gamma: Optional[oneflow._oneflow_internal.BlobDesc] = None,
    beta: Optional[oneflow._oneflow_internal.BlobDesc] = None,
    begin_norm_axis: int = 1,
    begin_params_axis: int = -1,
    epsilon: float = 1e-05,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Layer Normalization.

    Args:
        inputs (oneflow._oneflow_internal.BlobDesc): Input `Blob`.
        gamma (Optional[oneflow._oneflow_internal.BlobDesc]).
        beta (Optional[oneflow._oneflow_internal.BlobDesc]).
        begin_norm_axis (int, optional): An integer specifies which axis to normalize at first. Defaults to 1.
        begin_params_axis (int, optional):  An integer specifies which axis params at . Defaults to -1.
        epsilon (float, optional): A small float is added to avoid division by zero. Defaults to 1e-5.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: A normalized `Blob` with same shape of input.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def layer_norm_Job(x: tp.Numpy.Placeholder((1, 64, 128, 128))
        ) -> tp.Numpy:
            layer_norm = flow.nn.layer_norm(
                x,
                name="LayerNorm1"
            )
            return layer_norm


        x = np.random.randn(1, 64, 128, 128).astype(np.float32)
        out = layer_norm_Job(x)

        # out.shape (1, 64, 128, 128)

    """
    param_shape = inputs.shape[begin_params_axis:]
    if name is None:
        name = id_util.UniqueStr("LayerNorm_")
    if flow.current_scope().device_parallel_desc_symbol.device_tag == "cpu":
        if begin_norm_axis < 0:
            begin_norm_axis = begin_norm_axis + len(inputs.shape)
        reduce_axis = []
        for dim in range(len(inputs.shape)):
            if dim >= begin_norm_axis:
                reduce_axis.append(dim)
        (mean, variance) = flow.nn.moments(inputs, reduce_axis, keepdims=True)
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
        scale = False
        center = False
        if beta is not None:
            center = True
            op_builder.Input("beta", [beta])
        if gamma is not None:
            scale = True
            op_builder.Input("gamma", [gamma])
            op_builder.Output("normalized")
        op_builder.Attr("center", center)
        op_builder.Attr("scale", scale)
        op_builder.Attr("begin_norm_axis", begin_norm_axis)
        op_builder.Attr("begin_params_axis", begin_params_axis)
        op_builder.Attr("epsilon", epsilon)
        y = op_builder.Build().InferAndTryRun().RemoteBlobList()[0]
        return y
    else:
        raise NotImplementedError


def tf_conv2d(
    input: oneflow._oneflow_internal.BlobDesc,
    filters: oneflow._oneflow_internal.BlobDesc,
    strides: Union[int, Sequence[int]],
    padding: str,
    data_format: str = "NCHW",
    dilations: Optional[Union[int, Sequence[int]]] = None,
    groups: int = 1,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Computes a 2-D convolution given `input` and 4-D `filters` `Blob`.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): A `Blob` of rank at least 4.
        filters (oneflow._oneflow_internal.BlobDesc): A `Blob` with the same type as `input` and has the shape `[out_channels, in_channels//groups, filter_height, filter_width] for NCHW, or [out_channels, filter_height, filter_width, in_channels//groups] for NHWC`
        strides (Union[int, Sequence[int]]): An int or list of `ints` that has length `1`, or `2`. The stride of the sliding window for each dimension of `input`.
        padding (str): `"SAME"` or `"VALID"` indicating the type of padding algorithm to use, or a list indicating the explicit paddings at the start and end of each dimension.
        data_format (str, optional): `"NHWC"` or `"NCHW"`. Defaults to `"NCHW"`.
        dilations (Optional[Union[int, Sequence[int]]], optional): The dilation factor for each dimension of`input`. Defaults to None.
        groups (int, optional): int value greater than 0. Defaults to 1.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Raises:
        ValueError: strides must be an int or a list.
        ValueError: data_format must be "NHWC" or "NCHW".
        ValueError: dilations length must be 2 when passed as a list.
        ValueError: dilations must be an int or a list.
        ValueError: data_format NHWC not support groups > 1.
        ValueError: invalid data_format.
        ValueError: padding must be "SAME" or "VALID".

    Returns:
        oneflow._oneflow_internal.BlobDesc:  A `Blob` with the same type as `input` and the same outer batch shape.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        def conv2d(input, filters, kernel_size, strides, padding, name):
            input_shape = input.shape
            weight_initializer = flow.truncated_normal(0.1)
            weight_regularizer = flow.regularizers.l2(0.0005)
            weight_shape = (filters,
                            input_shape[1],
                            kernel_size[0],
                            kernel_size[1])

            weight = flow.get_variable(
                name + "-weight",
                shape=weight_shape,
                initializer=weight_initializer,
                regularizer=weight_regularizer,
            )
            return flow.nn.compat_conv2d(input, weight, strides, padding, name=name)


        @flow.global_function()
        def conv2d_Job(x: tp.Numpy.Placeholder((1, 64, 32, 32))
        ) -> tp.Numpy:
            conv = conv2d(x,
                        filters=128,
                        kernel_size=[3, 3],
                        strides=2,
                        padding='SAME',
                        name="Convlayer")
            return conv


        x = np.random.randn(1, 64, 32, 32).astype(np.float32)
        out = conv2d_Job(x)

        # out.shape (1, 128, 16, 16)

    """
    if padding.upper() == "SAME":
        padding = "SAME_UPPER"
    return flow.nn.conv2d(
        input, filters, strides, padding, None, data_format, dilations, groups, name
    )


def bias_add(
    value: oneflow._oneflow_internal.BlobDesc,
    bias: oneflow._oneflow_internal.BlobDesc,
    data_format: Optional[str] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator adds a bias to Blob.

    Args:
        value (oneflow._oneflow_internal.BlobDesc):  A `Blob`.
        bias (oneflow._oneflow_internal.BlobDesc): A 1-D `Blob` with size matching the channel dimension of value. And has the same type as value unless value is a quantized type.
        data_format (Optional[str], optional): A string. '`N...C'` or '`NC...'`. Defaults to None.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Raises:
        ValueError: ValueError if data format is unrecognized, if value has less than two dimensions with '`N..C'`/None data_format or value has less than three dimensions with '`NC..'` data_format, if bias is a vector, or if the size of bias does not match the size of the channel dimension of value.

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` with the same type as value.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def bias_add_Job(x: tp.Numpy.Placeholder((1, 64, 128, 128))
        ) -> tp.Numpy:
            bias_initializer = flow.truncated_normal(0.1)
            bias_regularizer = flow.regularizers.l2(0.0005)
            bias = flow.get_variable(
                    "Add_bias",
                    shape=(64,),
                    initializer=bias_initializer,
                    regularizer=bias_regularizer,
                )
            bias_out = flow.nn.bias_add(x, bias)
            return bias_out


        x = np.random.randn(1, 64, 128, 128).astype(np.float32)
        out = bias_add_Job(x)

        # out.shape (1, 64, 128, 128)

    """
    if name is None:
        name = id_util.UniqueStr("BiasAdd_")
    if data_format is None:
        bias_add_axis = 1
    elif data_format.startswith("NC"):
        bias_add_axis = 1
    elif data_format.startswith("N") and data_format.endswith("C"):
        bias_add_axis = len(value.shape) - 1
    else:
        raise ValueError("data_format must be of the form `N...C` or `NC...`")
    return (
        flow.user_op_builder(name)
        .Op("bias_add")
        .Input("a", [value])
        .Input("b", [bias])
        .Output("out")
        .Attr("axis", bias_add_axis)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def fused_bias_add_gelu(
    value: oneflow._oneflow_internal.BlobDesc,
    bias: oneflow._oneflow_internal.BlobDesc,
    data_format: Optional[str] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator fuse flow.nn.bias_add and flow.math.gelu operator.

    Args:
        value (oneflow._oneflow_internal.BlobDesc):  A `Blob`.
        bias (oneflow._oneflow_internal.BlobDesc): A 1-D `Blob` with size matching the channel dimension of value. And has the same type as value unless value is a quantized type.
        data_format (Optional[str], optional): A string. '`N...C'` or '`NC...'`. Defaults to None.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Raises:
        ValueError: ValueError if data format is unrecognized, if value has less than two dimensions with '`N..C'`/None data_format or value has less than three dimensions with '`NC..'` data_format, if bias is a vector, or if the size of bias does not match the size of the channel dimension of value.

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` with the same type as value.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def fused_bias_add_gelu_Job(x: tp.Numpy.Placeholder((1, 64, 128, 128))
        ) -> tp.Numpy:
            bias_initializer = flow.truncated_normal(0.1)
            bias_regularizer = flow.regularizers.l2(0.0005)
            bias = flow.get_variable(
                    "Add_bias",
                    shape=(64,),
                    initializer=bias_initializer,
                    regularizer=bias_regularizer,
                )
            out = flow.nn.fused_bias_add_gelu(x, bias)
            return out


        x = np.random.randn(1, 64, 128, 128).astype(np.float32)
        out = fused_bias_add_gelu_Job(x)

        # out.shape (1, 64, 128, 128)

    """
    if name is None:
        name = id_util.UniqueStr("FusedBiasAddGelu_")
    if data_format is None:
        bias_add_axis = 1
    elif data_format.startswith("NC"):
        bias_add_axis = 1
    elif data_format.startswith("N") and data_format.endswith("C"):
        bias_add_axis = len(value.shape) - 1
    else:
        raise ValueError("data_format must be of the form `N...C` or `NC...`")
    return (
        flow.user_op_builder(name)
        .Op("fused_bias_add_gelu")
        .Input("a", [value])
        .Input("b", [bias])
        .Output("out")
        .Attr("axis", bias_add_axis)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def fused_bias_add_dropout(
    value: oneflow._oneflow_internal.BlobDesc,
    bias: oneflow._oneflow_internal.BlobDesc,
    data_format: Optional[str] = None,
    rate: float = 0.0,
    noise_shape: Optional[oneflow._oneflow_internal.BlobDesc] = None,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator fuse flow.nn.bias_add and flow.nn.dropout operator.

    Args:
        value (oneflow._oneflow_internal.BlobDesc):  A `Blob`.
        bias (oneflow._oneflow_internal.BlobDesc): A 1-D `Blob` with size matching the channel dimension of value. And has the same type as value unless value is a quantized type.
        data_format (Optional[str], optional): A string. '`N...C'` or '`NC...'`. Defaults to None.
        rate (float): A scalar `Blob` with the same type as x. The probability that each element is dropped.
        noise_shape (Optional[oneflow._oneflow_internal.BlobDesc], optional):  optional: A 1-D `Blob`, representing the shape for randomly generated keep/drop flags. Defaults to None.Defaults to None.
        seed (Optional[int], optional):  Optional int value. Defaults to None.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Raises:
        ValueError: ValueError if data format is unrecognized, if value has less than two dimensions with '`N..C'`/None data_format or value has less than three dimensions with '`NC..'` data_format, if bias is a vector, or if the size of bias does not match the size of the channel dimension of value.

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` with the same type as value.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def fused_bias_add_dropout_Job(x: tp.Numpy.Placeholder((1, 64, 128, 128))
        ) -> tp.Numpy:
            bias_initializer = flow.truncated_normal(0.1)
            bias_regularizer = flow.regularizers.l2(0.0005)
            bias = flow.get_variable(
                    "Add_bias",
                    shape=(64,),
                    initializer=bias_initializer,
                    regularizer=bias_regularizer,
                )
            out = flow.nn.fused_bias_add_dropout(x, bias)
            return out


        x = np.random.randn(1, 64, 128, 128).astype(np.float32)
        out = fused_bias_add_dropout_Job(x)

        # out.shape (1, 64, 128, 128)

    """
    assert rate is not None and rate >= 0.0 and (rate < 1.0)
    if not flow.current_global_function_desc().IsTrainable() or rate == 0.0:
        return flow.nn.bias_add(value, bias, data_format, name)
    if name is None:
        name = id_util.UniqueStr("BiasAddDropout_")
    mask = flow.nn.random_mask_like(
        value, rate, seed, noise_shape, "%s-dropout_random_mask_like" % name
    )
    if data_format is None:
        bias_add_axis = 1
    elif data_format.startswith("NC"):
        bias_add_axis = 1
    elif data_format.startswith("N") and data_format.endswith("C"):
        bias_add_axis = len(value.shape) - 1
    else:
        raise ValueError("data_format must be of the form `N...C` or `NC...`")
    return (
        flow.user_op_builder(name)
        .Op("fused_bias_add_mask_scale")
        .Input("a", [value])
        .Input("b", [bias])
        .Input("mask", [mask])
        .Output("out")
        .Attr("axis", bias_add_axis)
        .Attr("scale", float(1.0 / (1.0 - rate)))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def max_pool1d(
    input: oneflow._oneflow_internal.BlobDesc,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Sequence[int]]],
    data_format: str = "NWC",
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Performs the 1d-max pooling on the input.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): A 3-D `Blob` of the format specified by data_format.
        ksize (Union[int, Sequence[int]]): An int or list of ints that has length 1 or 3. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1 or 3. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str):  '`VALID'` or '`SAME'`. The padding algorithm.
        data_format (str, optional):  An optional string from: '`NWC'`, '`NCW'`. Defaults to '`NWC'`.
        name (Optional[str], optional): This operator's name(optional).Defaults to None.

    Raises:
        NotImplementedError: TODO: fix cuDNN bugs in pooling_1d

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` of format specified by data_format. The max pooled output `Blob`.
    """
    raise NotImplementedError


def avg_pool1d(
    input: oneflow._oneflow_internal.BlobDesc,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Sequence[int]]],
    data_format: str = "NCW",
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Performs the average pooling on the input `Blob`.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): A 3-D `Blob` of the format specified by data_format.
        ksize (Union[int, Sequence[int]]): An int or list of ints that has length 1 or 3. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1 or 3. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'`.
        data_format (str, optional):  '`NWC'` or '`NCW'`. Defaults to '`NWC'`.
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Raises:
        NotImplementedError: TODO: fix cuDNN bugs in pooling_1d

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` of format specified by data_format. The max pooled output `Blob`.
    """
    raise NotImplementedError


def calc_pool_padding(padding, dhw_offset, ndims):
    if isinstance(padding, str):
        padding = "SAME_LOWER" if padding.upper() == "SAME" else padding
        assert padding.upper() in ["VALID", "SAME_LOWER", "SAME_UPPER"]
        padding_type = padding.lower()
        ndim_pads_list = [[0, 0]] * ndims
    elif isinstance(padding, (list, tuple)):
        padding_type = "customized"
        ndim_pads_list = get_ndim_pads_list(padding, dhw_offset, ndims)
    else:
        raise ValueError("padding must be str or a list.")
    return (padding_type, ndim_pads_list)


def max_pool2d(
    input: oneflow._oneflow_internal.BlobDesc,
    ksize: Union[int, IntPair],
    strides: Union[int, IntPair],
    padding: Union[str, Tuple[IntPair, IntPair, IntPair, IntPair]],
    data_format: str = "NCHW",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Performs the 2d-max pooling on the input `Blob`.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): A 4-D `Blob` of the format specified by data_format.
        ksize (Union[int, IntPair]): An int or list of ints that has length 1, 2. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, IntPair]): An int or list of ints that has length 1, 2. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'` or '`SAME_LOWER'` or '`SAME_UPPER'` or Tuple[IntPair, IntPair, IntPair, IntPair]`. The padding algorithm.
        data_format (str, optional): '`NHWC'`, '`NCHW'` or '`NCHW_VECT_C'`. Defaults to "NCHW".
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc:  A `Blob` of format specified by data_format. The max pooled output `Blob`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def maxpool2d_Job(x: tp.Numpy.Placeholder((1, 32, 128, 128))
        ) -> tp.Numpy:
            pool_out = flow.nn.max_pool2d(
                input=x,
                ksize=3,
                strides=2,
                padding='SAME',
                data_format='NCHW'
            )

            return pool_out


        x = np.random.randn(1, 32, 128, 128).astype(np.float32)
        out = maxpool2d_Job(x)

        # out.shape (1, 32, 64, 64)

    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("MaxPool2D_")
        )
        .Op("max_pool_2d")
        .Input("x", [input])
        .Output("y")
    )
    assert data_format in ["NHWC", "NCHW", "NCHW_VECT_C"]
    channel_pos = "channels_last" if data_format == "NHWC" else "channels_first"
    op.Attr("data_format", channel_pos)
    pool_size = _GetSequence(ksize, 2, "ksize")
    op.Attr("pool_size", pool_size)
    strides = _GetSequence(strides, 2, "strides")
    op.Attr("strides", strides)
    (padding_type, pads_list) = calc_pool_padding(
        padding, get_dhw_offset(channel_pos), 2
    )
    assert len(pads_list) == len(input.shape) - 2
    padding_before = [pad[0] for pad in pads_list]
    padding_after = [pad[1] for pad in pads_list]
    op.Attr("padding", padding_type)
    op.Attr("padding_before", padding_before)
    op.Attr("padding_after", padding_after)
    op.Attr("ceil_mode", ceil_mode)
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


def avg_pool2d(
    input: oneflow._oneflow_internal.BlobDesc,
    ksize: Union[int, IntPair],
    strides: Union[int, IntPair],
    padding: Union[str, Tuple[IntPair, IntPair, IntPair, IntPair]],
    data_format: str = "NCHW",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Performs the 2d-average pooling on the input.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): A 4-D `Blob` of shape [batch, height, width, channels].
        ksize (Union[int, IntPair]):  An int or list of ints that has length 1, 2. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, IntPair]): An int or list of ints that has length 1, 2. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'` or '`SAME_LOWER'` or '`SAME_UPPER'` or Tuple[IntPair, IntPair, IntPair, IntPair]. The padding algorithm.
        data_format (str, optional): '`NHWC'` or '`NCHW'`. Defaults to "NCHW".
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc:  A `Blob` with the same type as '`value'`. The average pooled output `Blob`.

    For example:

    .. code-block:: python

        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def avgpool2d_Job(x: tp.Numpy.Placeholder((1, 32, 128, 128))
        ) -> tp.Numpy:
            pool_out = flow.nn.avg_pool2d(
                input=x,
                ksize=3,
                strides=2,
                padding='SAME',
                data_format='NCHW'
            )

            return pool_out


        x = np.random.randn(1, 32, 128, 128).astype(np.float32)
        out = avgpool2d_Job(x)

        # out.shape (1, 32, 64, 64)

    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("AvgPool2D_")
        )
        .Op("avg_pool_2d")
        .Input("x", [input])
        .Output("y")
    )
    assert data_format in ["NHWC", "NCHW", "NCHW_VECT_C"]
    channel_pos = "channels_last" if data_format == "NHWC" else "channels_first"
    op.Attr("data_format", channel_pos)
    pool_size = _GetSequence(ksize, 2, "ksize")
    op.Attr("pool_size", pool_size)
    strides = _GetSequence(strides, 2, "strides")
    op.Attr("strides", strides)
    (padding_type, pads_list) = calc_pool_padding(
        padding, get_dhw_offset(channel_pos), 2
    )
    assert len(pads_list) == len(input.shape) - 2
    padding_before = [pad[0] for pad in pads_list]
    padding_after = [pad[1] for pad in pads_list]
    op.Attr("padding", padding_type)
    op.Attr("padding_before", padding_before)
    op.Attr("padding_after", padding_after)
    op.Attr("ceil_mode", ceil_mode)
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


def max_pool3d(
    input: oneflow._oneflow_internal.BlobDesc,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Sequence[int]]],
    data_format: str = "NCDHW",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Performs the 3d-max pooling on the input.

    Args:
        input (oneflow._oneflow_internal.BlobDesc):  A 5-D `Blob` of the format specified by data_format.
        ksize (Union[int, Sequence[int]]):  An int or list of ints that has length 1, 3 or 5. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1, 3 or 5. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'` or '`SAME_LOWER'` or '`SAME_UPPER'` or '`Sequence[Sequence[int]]'`.
        data_format (str, optional):   "NDHWC" or "NCDHW". Defaults to "NCDHW".
        name (Optional[str], optional): This operator's name(optional).

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` of format specified by data_format. The max pooled output `Blob`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def maxpool3d_Job(x: tp.Numpy.Placeholder((1, 32, 10, 128, 128))
        ) -> tp.Numpy:
            pool_out = flow.nn.max_pool3d(
                input=x,
                ksize=3,
                strides=2,
                padding='SAME',
                data_format='NCDHW'
            )

            return pool_out


        x = np.random.randn(1, 32, 10, 128, 128).astype(np.float32)
        out = maxpool3d_Job(x)

        # out.shape (1, 32, 5, 64, 64)

    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("MaxPool3D_")
        )
        .Op("max_pool_3d")
        .Input("x", [input])
        .Output("y")
    )
    assert data_format in ["NDHWC", "NCDHW"]
    channel_pos = "channels_last" if data_format == "NDHWC" else "channels_first"
    op.Attr("data_format", channel_pos)
    pool_size = _GetSequence(ksize, 3, "ksize")
    op.Attr("pool_size", pool_size)
    strides = _GetSequence(strides, 3, "strides")
    op.Attr("strides", strides)
    (padding_type, pads_list) = calc_pool_padding(
        padding, get_dhw_offset(channel_pos), 3
    )
    assert len(pads_list) == len(input.shape) - 2
    padding_before = [pad[0] for pad in pads_list]
    padding_after = [pad[1] for pad in pads_list]
    op.Attr("padding", padding_type)
    op.Attr("padding_before", padding_before)
    op.Attr("padding_after", padding_after)
    op.Attr("ceil_mode", ceil_mode)
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


def avg_pool3d(
    input: oneflow._oneflow_internal.BlobDesc,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Sequence[int]]],
    data_format: str = "NCDHW",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Performs the 3d-average pooling on the input.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): A 5-D `Blob` of shape [batch, height, width, channels].
        ksize (Union[int, Sequence[int]]): An int or list of ints that has length 1, 3 or 5. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1, 3 or 5. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'` or '`SAME_LOWER'` or '`SAME_UPPER or Sequence[Sequence[int]]'`.
        data_format (str, optional):  '`NDHWC'` or '`NCDHW'`. Defaults to "NCDHW".
        name (Optional[str], optional):  This operator's name(optional).Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` with the same type as value. The average pooled output `Blob`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def avgpool3d_Job(x: tp.Numpy.Placeholder((1, 32, 10, 128, 128))
        ) -> tp.Numpy:
            pool_out = flow.nn.avg_pool3d(
                input=x,
                ksize=3,
                strides=2,
                padding='SAME',
                data_format='NCDHW'
            )

            return pool_out


        x = np.random.randn(1, 32, 10, 128, 128).astype(np.float32)
        out = avgpool3d_Job(x)

        # out.shape (1, 32, 5, 64, 64)

    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("AvgPool3D_")
        )
        .Op("avg_pool_3d")
        .Input("x", [input])
        .Output("y")
    )
    assert data_format in ["NDHWC", "NCDHW"]
    channel_pos = "channels_last" if data_format == "NDHWC" else "channels_first"
    op.Attr("data_format", channel_pos)
    pool_size = _GetSequence(ksize, 3, "ksize")
    op.Attr("pool_size", pool_size)
    strides = _GetSequence(strides, 3, "strides")
    op.Attr("strides", strides)
    (padding_type, pads_list) = calc_pool_padding(
        padding, get_dhw_offset(channel_pos), 3
    )
    assert len(pads_list) == len(input.shape) - 2
    padding_before = [pad[0] for pad in pads_list]
    padding_after = [pad[1] for pad in pads_list]
    op.Attr("padding", padding_type)
    op.Attr("padding_before", padding_before)
    op.Attr("padding_after", padding_after)
    op.Attr("ceil_mode", ceil_mode)
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


def _softmax_need_transpose(x, axis):
    assert type(axis) is int
    dim_num = len(x.shape)
    assert dim_num >= 2
    if axis < 0:
        axis += dim_num
    assert axis >= 0
    assert axis < dim_num
    need_transpose = False
    permute = list(range(dim_num))
    if axis != dim_num - 1:
        need_transpose = True
        permute[axis] = permute[-1]
        permute[-1] = axis
    return (need_transpose, permute)


def softmax(
    logits: oneflow._oneflow_internal.BlobDesc,
    axis: Optional[int] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Computes softmax activations.

    For each element, we apply:

    .. math::
        S_i = \\frac{e^i}{\\sum_1^j e^j }

    Args:
        logits (oneflow._oneflow_internal.BlobDesc): A non-empty `Blob`.
        axis (Optional[int], optional): The dimension softmax would be performed on. Defaults to None.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc:  A `Blob` has the same type and shape as logits.

    Raises:
        InvalidArgumentError: if logits is empty or axis is beyond the last dimension of logits.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def softmax_Job(x: tp.Numpy.Placeholder((1, 5))
        ) -> tp.Numpy:
            softmax_out = flow.nn.softmax(x, axis=1)

            return softmax_out


        x = np.array([[1, 2, 1, 5, 4]]).astype(np.float32)
        out = softmax_Job(x)

        # out [[0.01259415 0.03423444 0.01259415 0.68761706 0.2529602 ]]

    """
    if axis is None:
        axis = -1
    (need_transpose, permute) = _softmax_need_transpose(logits, axis)
    if need_transpose:
        logits = flow.transpose(logits, perm=permute)
    out = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Softmax_")
        )
        .Op("softmax")
        .Input("in", [logits])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
    if need_transpose:
        out = flow.transpose(out, perm=permute)
    return out


def logsoftmax(
    logits: oneflow._oneflow_internal.BlobDesc,
    axis: Optional[int] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Computes logsoftmax activations.

    For each element, we apply:

    .. math::

        LogSoftmax(x_i) = Log(\\frac{e^i}{\\sum_1^j e^j })

    Args:
        logits (oneflow._oneflow_internal.BlobDesc): A non-empty `Blob`.
        axis (Optional[int], optional): The dimension logsoftmax would be performed on. Defaults to None.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc:  A `Blob` has the same type and shape as logits.

    Raises:
        InvalidArgumentError: if logits is empty or axis is beyond the last dimension of logits.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def logsoftmax_Job(x: tp.Numpy.Placeholder((1, 5))
        ) -> tp.Numpy:
            logsoftmax_out = flow.nn.logsoftmax(x, axis=1)
            return logsoftmax_out


        x = np.array([[1, 2, 1, 5, 4]]).astype(np.float32)
        out = logsoftmax_Job(x)

        # out [[-4.374523  -3.3745232 -4.374523  -0.3745232 -1.374523 ]]
    """
    if axis is None:
        axis = -1
    if name is None:
        name = id_util.UniqueStr("logsoftmax")
    return flow.math.log(
        flow.nn.softmax(logits, axis, name=name + "_softmax"), name=name + "_log"
    )


def softmax_grad(
    y: oneflow._oneflow_internal.BlobDesc,
    dy: oneflow._oneflow_internal.BlobDesc,
    axis: Optional[int] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Computes gradient of softmax activations.

    Args:
        y (oneflow._oneflow_internal.BlobDesc):  A `Blob` representing the softmax of x.
        dy (oneflow._oneflow_internal.BlobDesc):  gradient of y.
        axis (Optional[int], optional):  The dimension softmax would be performed on. Defaults to None.
        name (Optional[str], optional):  This operator's name(optional).

    Returns:
        oneflow._oneflow_internal.BlobDesc:  A `Blob` representing the gradient of x.
    """
    if axis is None:
        axis = -1
    (need_transpose, permute) = _softmax_need_transpose(y, axis)
    if need_transpose:
        y = flow.transpose(y, perm=permute)
        dy = flow.transpose(dy, perm=permute)
    dx = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Softmax_")
        )
        .Op("softmax_grad")
        .Input("y", [y])
        .Input("dy", [dy])
        .Output("dx")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
    if need_transpose:
        dx = flow.transpose(dx, perm=permute)
    return dx


def sparse_cross_entropy(
    labels: oneflow._oneflow_internal.BlobDesc,
    prediction: oneflow._oneflow_internal.BlobDesc,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Computes sparse cross entropy

    Args:
        labels (oneflow._oneflow_internal.BlobDesc): A `Blob` of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels and result). Each entry in labels must be an index in [0, num_classes).
        prediction (oneflow._oneflow_internal.BlobDesc): A `Blob` with the rank that is equal to the rank of the labels plus one.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` of the same shape as labels.

    Note:

        The labels data type should be `oneflow.int32`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def sparse_cross_entropy_Job(input: tp.Numpy.Placeholder((5, 2), dtype=flow.float32),
                                    labels: tp.Numpy.Placeholder((5,), dtype=flow.int32)
        ) -> tp.Numpy:
            loss = flow.nn.sparse_cross_entropy(labels=labels,
                                                prediction=input)
            return loss


        x = np.array([[0.3, 0.7],
                    [0.4, 0.6],
                    [0.5, 0.5],
                    [0.1, 0.9],
                    [0.2, 0.8]]).astype(np.float32)
        labels = np.array([0, 1, 1, 0, 1]).astype(np.int32)
        loss = sparse_cross_entropy_Job(x, labels)

        # out [1.2039728  0.5108256  0.6931472  2.3025851  0.22314353]

    """
    assert labels is not None
    assert prediction is not None
    if len(labels.shape) == len(prediction.shape):
        assert labels.shape[-1] == 1
        labels = flow.squeeze(labels, axis=[-1])
    else:
        assert len(labels.shape) == len(prediction.shape) - 1
    if prediction.distribute is oneflow._oneflow_internal.distribute.split(
        len(prediction.shape) - 1
    ):
        return (
            flow.user_op_builder(
                name if name is not None else id_util.UniqueStr("SparseCrossEntropyMs_")
            )
            .Op("sparse_cross_entropy_ms")
            .Input("prediction", [prediction])
            .Input("label", [labels])
            .Output("out")
            .Attr("depth", int(prediction.shape[-1]))
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
        return (
            flow.user_op_builder(
                name if name is not None else id_util.UniqueStr("SparseCrossEntropy_")
            )
            .Op("sparse_cross_entropy")
            .Input("prediction", [prediction])
            .Input("label", [labels])
            .Output("out")
            .Attr("depth", int(prediction.shape[-1]))
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )


def softmax_cross_entropy_with_logits(
    labels: oneflow._oneflow_internal.BlobDesc,
    logits: oneflow._oneflow_internal.BlobDesc,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Computes softmax cross entropy between logits and labels.

    Args:
        labels (oneflow._oneflow_internal.BlobDesc): Each vector along the class dimension should hold a valid probability distribution.
        logits (oneflow._oneflow_internal.BlobDesc): Per-label activations, typically a linear output. logits has same shape and dtype as labels.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` that contains the softmax cross entropy loss. Its type is the same as logits and its shape is the same as labels except that it does not have the last dimension of labels.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def softmax_cross_entropy_Job(input: tp.Numpy.Placeholder((3, 3), dtype=flow.float32),
                                    labels: tp.Numpy.Placeholder((3, 3), dtype=flow.float32)
        ) -> tp.Numpy:
            loss = flow.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=input)
            return loss


        x = np.array([[4, 1, 2],
                    [3, 2, 3],
                    [1, 5, 10]]).astype(np.float32)
        labels = np.array([[0.9, 0.05, 0.05],
                        [0.3, 0.4, 0.3],
                        [0.8, 0.1, 0.1]]).astype(np.float32)
        loss = softmax_cross_entropy_Job(x, labels)

        # out [0.73441553 1.1240788  1.4488925 ]

    """
    assert labels is not None
    assert logits is not None
    assert labels.shape == logits.shape
    assert labels.dtype == logits.dtype
    (prob, out) = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("SoftmaxCrossEntropy_")
        )
        .Op("softmax_cross_entropy")
        .Input("prediction", [logits])
        .Input("label", [labels])
        .Output("prob")
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return out


def sparse_softmax_cross_entropy_with_logits(
    labels: oneflow._oneflow_internal.BlobDesc,
    logits: oneflow._oneflow_internal.BlobDesc,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Computes sparse softmax cross entropy between logits and labels.

    Args:
        labels (oneflow._oneflow_internal.BlobDesc): `Blob` of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels and result). Each entry in labels must be an index in [0, num_classes).
        logits (oneflow._oneflow_internal.BlobDesc): Unscaled log probabilities of shape [d_0, d_1, ..., d_{r-1},num_classes].
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Raises:
        ValueError: If logits are scalars (need to have rank >= 1) or if the rank of the labels is not equal to the rank of the logits minus one.

    Returns:
        oneflow._oneflow_internal.BlobDesc:  A `Blob` of the same shape as labels and of the same type as logits with the softmax cross entropy loss.

    Note:

        The labels data type should be `oneflow.int32`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def sparse_softmax_cross_entropy_Job(input: tp.Numpy.Placeholder((3, 3), dtype=flow.float32),
                                             labels: tp.Numpy.Placeholder((3, ), dtype=flow.int32)
        ) -> tp.Numpy:
            loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                    logits=input)
            return loss


        x = np.array([[4, 1, 2],
                    [3, 2, 3],
                    [1, 5, 10]]).astype(np.float32)
        labels = np.array([0, 1, 2]).astype(np.int32)
        loss = sparse_softmax_cross_entropy_Job(x, labels)

        # out [0.65784633 1.2842525  0.5557927 ]

    """
    assert labels is not None
    assert logits is not None
    if len(labels.shape) == len(logits.shape):
        assert labels.shape[-1] == 1
        labels = flow.squeeze(labels, axis=[-1])
    else:
        assert len(labels.shape) == len(logits.shape) - 1
    if logits.distribute is oneflow._oneflow_internal.distribute.split(
        len(logits.shape) - 1
    ):
        (prob, out) = (
            flow.user_op_builder(
                name
                if name is not None
                else id_util.UniqueStr("SparseSoftmaxCrossEntropyMs_")
            )
            .Op("sparse_softmax_cross_entropy_ms")
            .Input("prediction", [logits])
            .Input("label", [labels])
            .Output("prob")
            .Output("out")
            .Attr("depth", int(logits.shape[-1]))
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()
        )
    else:
        (prob, out) = (
            flow.user_op_builder(
                name
                if name is not None
                else id_util.UniqueStr("SparseSoftmaxCrossEntropy_")
            )
            .Op("sparse_softmax_cross_entropy")
            .Input("prediction", [logits])
            .Input("label", [labels])
            .Output("prob")
            .Output("out")
            .Attr("depth", int(logits.shape[-1]))
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()
        )
    return out


def distributed_sparse_softmax_cross_entropy_with_logits(
    labels: oneflow._oneflow_internal.BlobDesc,
    logits: oneflow._oneflow_internal.BlobDesc,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    assert labels is not None
    assert logits is not None
    if len(labels.shape) == len(logits.shape):
        assert labels.shape[-1] == 1
        labels = flow.squeeze(labels, axis=[-1])
    else:
        assert len(labels.shape) == len(logits.shape) - 1
    (prob, out) = (
        flow.user_op_builder(
            name
            if name is not None
            else id_util.UniqueStr("DistributedSparseSoftmaxCrossEntropy_")
        )
        .Op("sparse_softmax_cross_entropy_ms")
        .Input("prediction", [logits])
        .Input("label", [labels])
        .Output("prob")
        .Output("out")
        .Attr("depth", int(logits.shape[-1]))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return out


def sigmoid_cross_entropy_with_logits(
    labels: oneflow._oneflow_internal.BlobDesc,
    logits: oneflow._oneflow_internal.BlobDesc,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Computes sigmoid cross entropy given logits.

    Args:
        labels (oneflow._oneflow_internal.BlobDesc): A `Blob` of the same type and shape as logits.
        logits (oneflow._oneflow_internal.BlobDesc): A `Blob` of type float.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc:   A `Blob` of the same shape as logits with the componentwise logistic losses.

    Raises:
        ValueError: If logits and labels do not have the same shape.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def sigmoid_cross_entropy_Job(input: tp.Numpy.Placeholder((3, 2), dtype=flow.float32),
                                    labels: tp.Numpy.Placeholder((3, 2), dtype=flow.float32)
        ) -> tp.Numpy:
            loss = flow.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                            logits=input)
            return loss


        x = np.array([[4, 1],
                    [3, 2],
                    [1, 5]]).astype(np.float32)
        labels = np.array([[0.7, 0.3],
                        [0.4, 0.6],
                        [0.2, 0.8]]).astype(np.float32)
        loss = sigmoid_cross_entropy_Job(x, labels)

        # out [[0.612735   0.90472794]
        #      [0.89778364 0.6990613 ]
        #      [0.97783387 0.51372755]]


    """
    assert labels is not None
    assert logits is not None
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("SigmoidCrossEntropy_")
        )
        .Op("sigmoid_cross_entropy")
        .Input("prediction", [logits])
        .Input("label", [labels])
        .Output("loss")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()[0]


def _GetSequence(value, n, name):
    """Formats value from input"""
    if value is None:
        value = [1]
    elif not isinstance(value, collections.Sized):
        value = [value]
    current_n = len(value)
    if current_n == 1:
        return list(value * n)
    elif current_n == n:
        return list(value)
    else:
        raise ValueError(
            "{} should be of length 1 or {} but was {}".format(name, n, current_n)
        )


def random_mask_like(
    like: oneflow._oneflow_internal.BlobDesc,
    rate: float,
    seed: Optional[int] = None,
    noise_shape: Optional[Sequence] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Random mask `Blob` with same shape as '`like'`.

    Args:
        like (oneflow._oneflow_internal.BlobDesc): A `Blob`.
        rate (float): A float value for the probability that each element is dropped.
        seed (Optional[int], optional): Optional, int value. Defaults to None.
        noise_shape (Optional[Sequence], optional): Optional, A 1-D `Blob`, representing the shape for randomly generated keep/drop flags. Defaults to None.
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: A random mask `Blob` of the same shape of `like`.

    Raises:
        ValueError: If rate is not in [0, 1). Rate=1 is not allowed.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def random_mask_like_Job(like: tp.Numpy.Placeholder((5, 5), dtype=flow.float32)
        ) -> tp.Numpy:

            return flow.nn.random_mask_like(like=like,
                                            rate=0.5)


        like = np.ones(shape=(5, 5)).astype(np.float32)
        random_mask = random_mask_like_Job(like)

        # out [[0 0 0 0 0]
        #      [1 1 1 0 0]
        #      [1 0 1 1 0]
        #      [0 0 0 0 1]
        #      [1 0 1 1 1]]

    """
    assert rate is not None and rate >= 0.0 and (rate < 1.0)
    if noise_shape is not None:
        assert 0, "noise_shape will be supported later."
        assert isinstance(noise_shape, (list, tuple))
    if seed is not None:
        assert name is not None
    if name is None:
        mask_op = (
            flow.user_op_builder(id_util.UniqueStr("RandomMaskLike_"))
            .Op("random_mask_like")
            .Input("like", [like])
            .Output("out")
            .Attr("rate", float(rate))
        )
        if seed is not None:
            mask_op.Attr("seed", seed)
        else:
            mask_op.Attr("seed", random.randint(-sys.maxsize, sys.maxsize))
        return mask_op.Build().InferAndTryRun().RemoteBlobList()[0]
    else:
        module = flow.find_or_create_module(
            name, lambda: RandomMaskLike(rate=rate, seed=seed, name=name)
        )
        return module(like)


class RandomMaskLike(module_util.Module):
    def __init__(self, rate: float, seed: Optional[int] = None, name: str = None):
        module_util.Module.__init__(self, name)
        if seed is None:
            seed = random.randint(-sys.maxsize, sys.maxsize)
        self.op_module_builder = (
            flow.user_op_module_builder("random_mask_like")
            .InputSize("like", 1)
            .Output("out")
            .Attr("rate", float(rate))
            .Attr("seed", seed)
            .CheckAndComplete()
        )
        self.op_module_builder.user_op_module.InitOpKernel()

    def forward(self, like: oneflow._oneflow_internal.BlobDesc):
        if self.call_seq_no == 0:
            name = self.module_name
        else:
            name = id_util.UniqueStr("RandomMaskLike_")
        return (
            self.op_module_builder.OpName(name)
            .Input("like", [like])
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )


def dropout(
    x: oneflow._oneflow_internal.BlobDesc,
    rate: float,
    noise_shape: Optional[oneflow._oneflow_internal.BlobDesc] = None,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """For preventing overfitting, randomly set elements to zero.

    Args:
        x (oneflow._oneflow_internal.BlobDesc): A floating point `Blob`.
        rate (float): A scalar `Blob` with the same type as x. The probability that each element is dropped.
        noise_shape (Optional[oneflow._oneflow_internal.BlobDesc], optional):  optional: A 1-D `Blob`, representing the shape for randomly generated keep/drop flags. Defaults to None.Defaults to None.
        seed (Optional[int], optional):  Optional int value. Defaults to None.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc:   A `Blob` of the same shape of x.

    Raises:
        ValueError: If rate is not in [0, 1) or if x is not a floating point `Blob`. Rate=1 is not allowed.

    For example:

    .. code-block:: python

        import oneflow as flow


        def lenet(data, train=False):
            initializer = flow.truncated_normal(0.1)
            conv1 = flow.layers.conv2d(
                data,
                32,
                5,
                padding="SAME",
                activation=flow.nn.relu,
                name="conv1",
                kernel_initializer=initializer,
            )
            pool1 = flow.nn.max_pool2d(
                conv1, ksize=2, strides=2, padding="SAME", name="pool1", data_format="NCHW"
            )
            conv2 = flow.layers.conv2d(
                pool1,
                64,
                5,
                padding="SAME",
                activation=flow.nn.relu,
                name="conv2",
                kernel_initializer=initializer,
            )
            pool2 = flow.nn.max_pool2d(
                conv2, ksize=2, strides=2, padding="SAME", name="pool2", data_format="NCHW"
            )
            reshape = flow.reshape(pool2, [pool2.shape[0], -1])
            hidden = flow.layers.dense(
                reshape,
                512,
                activation=flow.nn.relu,
                kernel_initializer=initializer,
                name="dense1",
            )
            if train:
                hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")

            return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="dense2")

    """
    assert rate is not None and rate >= 0.0 and (rate < 1.0)
    if not flow.current_global_function_desc().IsTrainable() or rate == 0.0:
        return x
    if seed is not None:
        assert name is not None
    if name is None:
        name = id_util.UniqueStr("Dropout_")
    mask = random_mask_like(
        x, rate, seed, noise_shape, "%s-dropout_random_mask_like" % name
    )
    return (
        flow.user_op_builder(name)
        .Op("dropout")
        .Input("in", [x])
        .Input("mask", [mask])
        .Output("out")
        .Attr("scale", float(1.0 / (1.0 - rate)))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def deconv2d(
    value: Optional[oneflow._oneflow_internal.BlobDesc] = None,
    filter: Optional[oneflow._oneflow_internal.BlobDesc] = None,
    output_shape: Tuple[int, int, int, int] = None,
    strides: Optional[Union[int, Sequence[int]]] = None,
    padding: str = "VALID",
    data_format: str = "NCHW",
    name: Optional[str] = None,
    input: Optional[oneflow._oneflow_internal.BlobDesc] = None,
    filters: Optional[oneflow._oneflow_internal.BlobDesc] = None,
    dilations: Optional[Union[int, Sequence[int]]] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """2d transposed convolution.

    Args:
        value (Optional[oneflow._oneflow_internal.BlobDesc], optional):   4-d `Blob`. Defaults to None.
        filter (Optional[oneflow._oneflow_internal.BlobDesc], optional): Filter of transposed convolution, usually a variable. Defaults to None.
        output_shape (Tuple[int, int, int, int]): A 1-D `Blob` representing the output shape of the deconvolution op. Defaults to None.
        strides (Optional[Union[int, Sequence[int]]], optional): `int` or `int list`. Defaults to None.
        padding (str, optional):  `'VALID'` or `'SAME'`. Defaults to "VALID".
        data_format (str, optional): `'NHWC'` or `'NCHW'`. Defaults to "NCHW".
        name (Optional[str], optional): This operator's name(optional). Defaults to None.
        input (Optional[oneflow._oneflow_internal.BlobDesc], optional): Alias for value. Defaults to None.
        filters (Optional[oneflow._oneflow_internal.BlobDesc], optional): Alias for filter. Defaults to None.
        dilations (Optional[Union[int, Sequence[int]]], optional): The dilation factor for each dimension of input. Defaults to None.

    Raises:
        ValueError: shapes of `filter` and `input` must match.
        ValueError: dilations must be an int or a list.
        ValueError: data_format must be "NHWC" or "NCHW".
        ValueError: padding must be "SAME" or "VALID".

    Returns:
        oneflow._oneflow_internal.BlobDesc: A `Blob` with the same type as `value`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        def deconv2d(input, filters, kernel_size, strides, padding, name):
            input_shape = input.shape
            weight_initializer = flow.truncated_normal(0.1)
            weight_regularizer = flow.regularizers.l2(0.0005)
            weight_shape = (filters,
                            input_shape[1],
                            kernel_size[0],
                            kernel_size[1])

            weight = flow.get_variable(
                name + "-weight",
                shape=weight_shape,
                initializer=weight_initializer,
                regularizer=weight_regularizer,
            )
            return flow.nn.conv2d_transpose(value=input,
                                            output_shape=(1, 32, 64, 64),
                                            filter=weight,
                                            strides=strides,
                                            padding=padding,
                                            name=name)


        @flow.global_function()
        def deconv2d_Job(x: tp.Numpy.Placeholder((1, 32, 32, 32),)
        ) -> tp.Numpy:
            deconv = deconv2d(x,
                            filters=32,
                            kernel_size=[3, 3],
                            strides=2,
                            padding='SAME',
                            name="Convlayer")
            return deconv


        x = np.random.randn(1, 32, 32, 32).astype(np.float32)
        out = deconv2d_Job(x)

        # out.shape (1, 32, 64, 64)

    """
    assert (value is not None) ^ (
        input is not None
    ), "only one of `input` and `value` could be not None"
    assert (filter is not None) ^ (
        filters is not None
    ), "only one of `filter` and `filters` could be not None"
    filters = filters or filter
    input = input or value
    NDims = 2
    assert len(input.shape) == 2 + NDims
    assert len(filters.shape) == 2 + NDims
    assert len(output_shape) == 2 + NDims
    assert output_shape[0] == input.shape[0]
    if dilations is None:
        dilations = [1, 1]
    elif isinstance(dilations, (list, tuple)):
        assert len(dilations) == 2, ValueError(
            "dilations length must be 2 when passed as a list."
        )
    elif isinstance(dilations, int):
        dilations = [dilations, dilations]
    else:
        raise ValueError("dilations must be an int or a list.")
    if data_format.upper() == "NCHW":
        input_shape = input.shape[2:]
        kernel_size = filters.shape[2:4]
        channels = filters.shape[1]
        assert output_shape[1] == channels
        output_shape = output_shape[2:4]
    elif data_format.upper() == "NHWC":
        input_shape = input.shape[1:3]
        kernel_size = filters.shape[-3:-1]
        channels = filters.shape[3]
        assert output_shape[3] == channels
        output_shape = output_shape[1:3]
        assert dilations == [1, 1], ValueError(
            "dialtions must be 1 when data format is NHWC "
        )
    else:
        raise ValueError('data_format must be "NHWC" or "NCHW".')
    channel_pos = "channels_first" if data_format.startswith("NC") else "channels_last"
    if isinstance(strides, (list, tuple)):
        assert len(strides) == NDims, ValueError(
            "strides length must be 2 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")
    output_padding = [0] * NDims
    padding_needed = [0] * NDims
    if padding.upper() == "VALID":
        for i in range(NDims):
            effective_filter_size = (kernel_size[i] - 1) * dilations[i] + 1
            assert (output_shape[i] + strides[i] - effective_filter_size) // strides[
                i
            ] == input_shape[i]
            tmp_output_shape = (input_shape[i] - 1) * strides[i] + effective_filter_size
            output_padding[i] = output_shape[i] - tmp_output_shape
    elif padding.upper() == "SAME":
        padding_left = [0] * NDims
        padding_right = [0] * NDims
        for i in range(NDims):
            assert (output_shape[i] + strides[i] - 1) // strides[i] == input_shape[i]
            effective_filter_size = (kernel_size[i] - 1) * dilations[i] + 1
            padding_needed[i] = max(
                0,
                (input_shape[i] - 1) * strides[i]
                + effective_filter_size
                - output_shape[i],
            )
            tmp_output_shape = (
                (input_shape[i] - 1) * strides[i]
                + effective_filter_size
                - padding_needed[i]
            )
            output_padding[i] = output_shape[i] - tmp_output_shape
            padding_left[i] = padding_needed[i] // 2
            padding_right[i] = padding_needed[i] - padding_needed[i] // 2
    else:
        raise ValueError('padding must be "SAME" or "VALID".')
    if padding.upper() == "SAME" and padding_left != padding_right:
        assert data_format.upper() == "NCHW"
        padding_before = [0] * NDims
        input = (
            flow.user_op_builder(
                name if name is not None else id_util.UniqueStr("Deconv2d_")
            )
            .Op("deconv2d")
            .Input("in", [input])
            .Input("weight", [filters])
            .Output("out")
            .Attr("filters", channels)
            .Attr("padding_before", padding_before)
            .Attr("data_format", channel_pos)
            .Attr("kernel_size", kernel_size)
            .Attr("strides", strides)
            .Attr("dilation_rate", dilations)
            .Attr("output_padding", output_padding)
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
        return flow.pad_grad(
            input,
            [
                (0, 0),
                (0, 0),
                (padding_left[0], padding_right[0]),
                (padding_left[1], padding_right[1]),
            ],
            name=name + "_pad_grad" if name is not None else None,
        )
    assert len(padding_needed) == len(input.shape) - 2
    padding_before = []
    for pad in padding_needed:
        assert pad % 2 == 0
        padding_before.append(pad // 2)
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Deconv2d_")
        )
        .Op("deconv2d")
        .Input("in", [input])
        .Input("weight", [filters])
        .Output("out")
        .Attr("filters", channels)
        .Attr("padding_before", padding_before)
        .Attr("data_format", channel_pos)
        .Attr("kernel_size", kernel_size)
        .Attr("strides", strides)
        .Attr("dilation_rate", dilations)
        .Attr("output_padding", output_padding)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def deconv2d_torch(
    value=None,
    filter=None,
    output_padding=None,
    strides=None,
    padding_needed=None,
    data_format="NCHW",
    name=None,
    input=None,
    filters=None,
    dilations=None,
):
    assert (value is not None) ^ (
        input is not None
    ), "only one of `input` and `value` could be not None"
    assert (filter is not None) ^ (
        filters is not None
    ), "only one of `filter` and `filters` could be not None"
    filters = filters or filter
    input = input or value
    NDims = 2
    assert len(input.shape) == 2 + NDims
    assert len(filters.shape) == 2 + NDims
    if dilations is None:
        dilations = [1, 1]
    elif isinstance(dilations, (list, tuple)):
        assert len(dilations) == 2, ValueError(
            "dilations length must be 2 when passed as a list."
        )
    elif isinstance(dilations, int):
        dilations = [dilations, dilations]
    else:
        raise ValueError("dilations must be an int or a list.")
    if data_format.upper() == "NCHW":
        input_shape = input.shape[2:]
        kernel_size = filters.shape[2:4]
        channels = filters.shape[1]
    elif data_format.upper() == "NHWC":
        input_shape = input.shape[1:3]
        kernel_size = filters.shape[-3:-1]
        channels = filters.shape[3]
        assert dilations == [1, 1], ValueError(
            "dialtions must be 1 when data format is NHWC "
        )
    else:
        raise ValueError('data_format must be "NHWC" or "NCHW".')
    channel_pos = "channels_first" if data_format.startswith("NC") else "channels_last"
    if isinstance(strides, (list, tuple)):
        assert len(strides) == NDims, ValueError(
            "strides length must be 2 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")
    assert len(padding_needed) == len(input.shape) - 2
    padding_before = []
    for pad in padding_needed:
        assert pad % 2 == 0
        padding_before.append(pad // 2)
    if output_padding is None:
        output_padding = (0, 0)
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Deconv2d_")
        )
        .Op("deconv2d")
        .Input("in", [input])
        .Input("weight", [filters])
        .Output("out")
        .Attr("filters", channels)
        .Attr("padding_before", padding_before)
        .Attr("data_format", channel_pos)
        .Attr("kernel_size", kernel_size)
        .Attr("strides", strides)
        .Attr("dilation_rate", dilations)
        .Attr("output_padding", output_padding)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def leaky_relu(
    x: oneflow._oneflow_internal.BlobDesc,
    alpha: float = 0.2,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Leaky ReLU activation.

    .. math::
        out = max(x, alpha*x)

    Args:
        x (oneflow._oneflow_internal.BlobDesc):  A `Blob` representing preactivation values.
        alpha (float, optional): Slope of the activation function at x < 0 with float type. Default value is 0.2.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The activation `Blob`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def leaky_relu_Job(x: tp.Numpy.Placeholder((5, ),)
        ) -> tp.Numpy:
            leaky_relu = flow.nn.leaky_relu(x, alpha=0.2)

            return leaky_relu


        x = np.array([-10, -5, 0, 5, 10]).astype(np.float32)
        out = leaky_relu_Job(x)

        # out [-2. -1.  0.  5. 10.]

    """
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("LeakyRelu_")
        )
        .Op("leaky_relu")
        .Input("x", [x])
        .Output("y")
        .Attr("alpha", float(alpha))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def elu(
    x: oneflow._oneflow_internal.BlobDesc,
    alpha: float = 1.0,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """The ELU activation.

    The formula is:

    .. math::

        \\text{ELU}(x) = \\begin{cases}
				x & \\text{ if } x \\gt 0  \\\\
                \\alpha*(exp(x)-1) & \\text{ if } x \\le 0 \\\\
    		    \\end{cases}

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        @flow.global_function()
        def elu_job(x: tp.Numpy.Placeholder(shape=(3, )))->tp.Numpy:
            return flow.nn.elu(x, alpha=1.0)


        x = np.array([-3.5, 1, 3.5]).astype(np.float32)
        out = elu_job(x)

        # output [-0.9698026  1.         3.5      ]

    Args:
        x (oneflow._oneflow_internal.BlobDesc): The input Tensor.
        alpha (float, optional): The `alpha` value for the ELU formula. Defaults to 1.0.
        name (Optional[str], optional): The name for the operator. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The activated Tensor.
    """
    alpha = float(alpha)
    if name is None:
        name = id_util.UniqueStr("Elu_")
    return (
        flow.user_op_builder(name)
        .Op("elu")
        .Input("in", [x])
        .Output("out")
        .Attr("alpha", alpha)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def hard_sigmoid(
    x: oneflow._oneflow_internal.BlobDesc, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    """The Hardsigmoid activation.

    The formula is:

    .. math::

        \\text{Hardsigmoid}(x) = \\begin{cases}
            0 & \\text{ if } x \\le -3  \\\\
            1 & \\text{ if } x \\ge +3 \\\\
            \\frac{x}{6} + \\frac{1}{2} & \\text{ otherwise } \\\\
        \\end{cases}

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        @flow.global_function()
        def hardsigmoid_job(x: tp.Numpy.Placeholder(shape=(3, )))->tp.Numpy:
            out = flow.nn.hardsigmoid(x)

            return out


        x = np.array([-3.1, 0, 3.3]).astype(np.float32)
        out = hardsigmoid_job(x)

        # output [0.  0.5 1. ]

    Args:
        x (oneflow._oneflow_internal.BlobDesc): The input Tensor.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The activated Tensor.
    """
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("HardSigmoid_")
        )
        .Op("hardsigmoid")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def mish(
    x: oneflow._oneflow_internal.BlobDesc, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    """The Mish activation function.

    The equation is:

    .. math::

        out = x*tanh(ln(1+e^x))

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        @flow.global_function()
        def mish_job(x: tp.Numpy.Placeholder(shape=(5, )))->tp.Numpy:
            return flow.nn.mish(x)


        x = np.array([-0.5, 0, 0.5, 1.0, 1.5]).astype(np.float32)
        out = mish_job(x)

    Args:
        x (oneflow._oneflow_internal.BlobDesc): The input Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob.
    """
    if name is None:
        name = id_util.UniqueStr("Mish_")
    return x * flow.math.tanh(
        flow.math.softplus(x, name=name + "softplus"), name=name + "tanh"
    )


def swish(
    x: oneflow._oneflow_internal.BlobDesc, beta: float = 1.0, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    """The Swish activation function.

    The equation is:

    .. math::

        out = x * sigmoid(\\beta*x)

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        @flow.global_function()
        def swish_job(x: tp.Numpy.Placeholder(shape=(5, )))->tp.Numpy:
            return flow.nn.swish(x)


        x = np.array([-0.5, 0, 0.5, 1, 1.5]).astype(np.float32)


        out = swish_job(x)
        # output [-0.18877034  0.          0.31122968  0.7310586   1.2263618 ]

    Args:
        x (oneflow._oneflow_internal.BlobDesc): The input Blob.
        beta (float, optional): The smooth factor. Defaults to 1.0.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob.
    """
    if name is None:
        name = id_util.UniqueStr("Swish_")
    return x * flow.math.sigmoid(beta * x, name=name + "_sigmoid")


def hardswish(
    x: oneflow._oneflow_internal.BlobDesc, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    """The Hardswish activation.

    The formula is:

    .. math::

        \\text{Hardswish}(x) = \\begin{cases}
            0 & \\text{ if } x \\le -3  \\\\
            x & \\text{ if } x \\ge +3 \\\\
            x*(x+3)/6 & \\text{ otherwise } \\\\
        \\end{cases}

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        @flow.global_function()
        def hardswish_job(x: tp.Numpy.Placeholder(shape=(3, )))->tp.Numpy:
            return flow.nn.hardswish(x)


        x = np.array([-3.5, 1, 3.5]).astype(np.float32)
        out = hardswish_job(x)

        # output [0.        0.6666667 3.5      ]

    Args:
        x (oneflow._oneflow_internal.BlobDesc): The input Tensor.
        name (Optional[str], optional): The name for the operation. Defaults to None.
    Returns:
        oneflow._oneflow_internal.BlobDesc: The activated Tensor.
    """
    if name is None:
        name = id_util.UniqueStr("HardSwish_")
    return (
        flow.user_op_builder(name)
        .Op("hardswish")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def hardtanh(
    x: oneflow._oneflow_internal.BlobDesc,
    min_val: float = -1.0,
    max_val: float = 1.0,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """The Hardtanh activation.

    The equation is:

    .. math::

        \\text{HardTanh}(x) = \\begin{cases}
            max\\_val & \\text{ if } x > max\\_val \\\\
            -min\\_val & \\text{ if } x < min\\_val \\\\
            x & \\text{ otherwise } \\\\
        \\end{cases}

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np

        @flow.global_function()
        def hardtanh_job(x: tp.Numpy.Placeholder(shape=(2, 3)))->tp.Numpy:
            return flow.nn.hardtanh(x, min_val=-1.25, max_val=1.2)


        x = np.array([[-1.5, -1.1, 0.6],
                    [1.2, 1.3, 1.5]]).astype(np.float32)
        out = hardtanh_job(x)

        # output [[-1.25 -1.1   0.6 ]
        #         [ 1.2   1.2   1.2 ]]

    Args:
        x (oneflow._oneflow_internal.BlobDesc): The input Tensor.
        min_val (float, optional): The minimum value of the linear region range. Defaults to -1.
        max_val (float, optional): The maximum value of the linear region range. Defaults to 1.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The activated tensor.
    """
    if name is None:
        name = id_util.UniqueStr("Hardtanh_")
    min_val = float(min_val)
    max_val = float(max_val)
    assert min_val < max_val, "max_val should be larger than min_val"
    return (
        flow.user_op_builder(name)
        .Op("hardtanh")
        .Input("in", [x])
        .Attr("min_val", min_val)
        .Attr("max_val", max_val)
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def relu6(
    x: oneflow._oneflow_internal.BlobDesc, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    """Relu6 activation, it clips the value around (0, 6).

    The equation is:

    .. math::

        \\text{Relu6}(x) = \\begin{cases}
            6 & \\text{ if } x > 6 \\\\
            0 & \\text{ if } x < 0 \\\\
            x & \\text{ otherwise } \\\\
        \\end{cases}

    For example:

    .. code-block::

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        @flow.global_function()
        def relu6_job(x: tp.Numpy.Placeholder(shape=(2, 3)))->tp.Numpy:
            return flow.nn.relu6(x)

        x = np.array([[-1, -0.5, 0.0],
                      [0.5, 6.0, 7]]).astype(np.float32)

        out = relu6_job(x)

        # output [[0.  0.  0. ]
        #         [0.5 6.  6. ]]

    Args:
        x (oneflow._oneflow_internal.BlobDesc): The input Tensor.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The activated Tensor.
    """
    if name is None:
        name = id_util.UniqueStr("Relu6_")
    return flow.nn.hardtanh(x, min_val=0.0, max_val=6.0, name=name)


def pixel_shufflev2(
    input: oneflow._oneflow_internal.BlobDesc,
    h_upscale_factor: int,
    w_upscale_factor: int,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator is similar to `oneflow.nn.PixelShuffle`. The difference is that in
    `oneflow.nn.PixelShuffle`, the upscale factor of height and width is the same. But in
    `oneflow.nn.PixelShufflev2`, you can set different upscale factor for height and width.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): The input Blob.
        h_upscale_factor (int): The upscale factor of height.
        w_upscale_factor (int): The upscale factor of width.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        @flow.global_function()
        def PixelShufflev2Job(input: tp.Numpy.Placeholder(shape=(3, 16, 2, 4), dtype=flow.float32))->tp.Numpy:
            out = flow.nn.PixelShufflev2(input, h_upscale_factor=2, w_upscale_factor=4)

            return out

        input = np.random.uniform(size=(3, 16, 2, 4)).astype(np.float32)
        out = PixelShuffleJob(input)

        # out.shape (3, 2, 4, 16)

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob.
    """
    assert (
        h_upscale_factor > 0 and w_upscale_factor > 0
    ), "The scale factor of height and width must larger than zero"
    assert len(input.shape) == 4, "Only Accept 4D Blob"
    (_batch, _channel, _height, _width) = input.shape
    assert (
        _channel % (h_upscale_factor * w_upscale_factor) == 0
    ), "The channels of input tensor must be divisible by (h_upscale_factor * w_upscale_factor)"
    if name is None:
        name = id_util.UniqueStr("PixelShufflev2")
    _new_c = int(_channel / (h_upscale_factor * w_upscale_factor))
    out = flow.reshape(
        input,
        [_batch, _new_c, h_upscale_factor * w_upscale_factor, _height, _width],
        name=name + "_reshape1",
    )
    out = flow.reshape(
        out,
        [_batch, _new_c, h_upscale_factor, w_upscale_factor, _height, _width],
        name=name + "_reshape2",
    )
    out = flow.transpose(out, [0, 1, 4, 2, 5, 3], name=name + "_transpose")
    out = flow.reshape(
        out,
        [_batch, _new_c, _height * h_upscale_factor, _width * w_upscale_factor],
        name=name + "_reshape3",
    )
    return out
