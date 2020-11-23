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

import collections
import os
import sys
import random
from typing import Union, Optional, Sequence, Tuple, List
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.module as module_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
from oneflow.python.oneflow_export import oneflow_export

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
                input_shape[dhw_offset + i], kernel_sizes[i], dilations[i], strides[i],
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
            return inputs, return_pads_list
        else:
            if is_dynamic:
                return_pads_list = [[0, 0]] * ndims
                inputs = flow.same_padding(
                    inputs,
                    padding.lower(),
                    data_format=data_format,
                    kernel_size=kernel_sizes,
                    strides=strides,
                    dilation_rate=dilations,
                )
                return inputs, return_pads_list
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
        return inputs, ndim_pads_list
    else:
        pad_op_list = [[0, 0]] * (ndims + 2)
        for i in range(ndims):
            pad_op_list[dhw_offset + i] = ndim_pads_list[i]
        inputs = flow.pad(inputs, paddings=pad_op_list)
        return_pads_list = [[0, 0]] * ndims
        return inputs, return_pads_list


@oneflow_export("nn.conv1d")
def conv1d(
    input: remote_blob_util.BlobDef,
    filters: remote_blob_util.BlobDef,
    strides: Union[int, Tuple[int]],
    padding: Union[str, Tuple[IntPair, IntPair, IntPair]],
    data_format: str = "NCW",
    dilations: Optional[Union[int, Tuple[int]]] = None,
    groups: int = 1,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""1D convolution layer.

    Args:
        input (remote_blob_util.BlobDef): A 3D input `Blob`. [batch_num, channel, width]
        filters (remote_blob_util.BlobDef): A `Blob` with the same type as `input` and has the shape [out_channels, in_channels//groups, filter_width] for `NCW`, or [out_channels, filter_width, in_channels//groups] for `NWC`
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
        remote_blob_util.BlobDef: A `Blob` with the same type as `input` and the same outer batch shape.
    
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
    else:
        if isinstance(dilations, (list, tuple)):
            assert len(dilations) == 1, ValueError(
                "dilations length must be 1 when passed as a list."
            )
        elif isinstance(dilations, int):
            dilations = [dilations]
        else:
            raise ValueError("dilations must be an int or a list.")

    if channel_pos == "channels_first":
        kernel_size_list = filters.shape[2:3]
    elif channel_pos == "channels_last":
        kernel_size_list = filters.shape[-2:-1]
    else:
        raise ValueError("invalid data_format")
    assert isinstance(kernel_size_list, tuple)
    assert isinstance(groups, int)
    assert groups > 0
    if groups > 1:
        if data_format.upper() == "NCW":
            assert groups <= filters.shape[0]
            assert filters.shape[0] % groups == 0
            assert groups <= input.shape[1]
            assert input.shape[1] % groups == 0
            assert filters.shape[1] == input.shape[1] // groups
        elif data_format.upper() == "NWC":
            raise ValueError("data_format NWC not support groups > 1")
        else:
            raise ValueError("invalid data_format")
    inputs, pads_list = calc_conv_padding(
        input, padding, data_format.upper(), kernel_size_list, dilations, strides,
    )
    assert len(pads_list) == len(inputs.shape) - 2
    padding_before = [pad[0] for pad in pads_list]

    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Conv1d_"))
        .Op("conv1d")
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
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


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


@oneflow_export("nn.conv2d")
def conv2d(
    input: remote_blob_util.BlobDef,
    filters: remote_blob_util.BlobDef,
    strides: Union[int, IntPair],
    padding: Union[str, Tuple[IntPair, IntPair, IntPair, IntPair]],
    data_format: str = "NCHW",
    dilations: Optional[Union[int, IntPair]] = None,
    groups: int = 1,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""2D convolution layer.

    Args:
        input (remote_blob_util.BlobDef): A 4D input `Blob`. [batch_num, channel, height, width]
        filters (remote_blob_util.BlobDef): A `Blob` with the same type as `input` and has the shape `[out_channels, in_channels//groups, filter_height, filter_width] for NCHW, or [out_channels, filter_height, filter_width, in_channels//groups] for NHWC`
        strides (Union[int, IntPair]): An int or list of `ints` that has length `2`. The stride of the sliding window for each dimension of `input`.
        padding (Union[str, Tuple[IntPair, IntPair, IntPair, IntPair]]): padding: `string` `"SAME"` or `"SAME_LOWER"` or `"SAME_UPPER"` or `"VALID" or Tuple[IntPair, IntPair, IntPair, IntPair]` indicating the type of padding algorithm to use, or a list indicating the explicit paddings at the start and end of each dimension.
        data_format (str, optional): `"NHWC" or "NCHW"`. Defaults to `"NCHW"`.
        dilations (Optional[Union[int, IntPair]], optional): An int or list of `ints` that has length `2`. The dilation factor for each dimension of `input`. Defaults to None.
        groups (int, optional): int value greater than 0. Defaults to 1.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Raises:
        ValueError: strides must be an int or a list.
        ValueError: padding must be "SAME" or `"SAME_LOWER" or "SAME_UPPER" or "VALID" or Tuple[IntPair, IntPair, IntPair, IntPair].
        ValueError: data_format must be "NHWC" or "NCHW".
        ValueError: dilations must be an int or a list.
        ValueError: invalid data_format.
        ValueError: data_format NHWC not support groups > 1
        ValueError: invalid data_format.

    Returns:
        remote_blob_util.BlobDef: A `Blob` with the same type as `input` and the same outer batch shape.
    
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
    else:
        if isinstance(dilations, (list, tuple)):
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
    inputs, pads_list = calc_conv_padding(
        input, padding, data_format.upper(), kernel_size_list, dilations, strides,
    )
    assert len(pads_list) == len(inputs.shape) - 2
    padding_before = [pad[0] for pad in pads_list]

    assert groups <= filters.shape[filter_out_axis]
    assert filters.shape[filter_out_axis] % groups == 0
    assert groups <= inputs.shape[in_channel_axis]
    assert inputs.shape[in_channel_axis] % groups == 0
    assert filters.shape[filter_in_axis] == inputs.shape[in_channel_axis] // groups

    if (
        groups > 1
        and flow.current_scope().device_parallel_desc_symbol.device_tag == "cpu"
    ):
        in_split_list = ConvUtil.split(inputs, axis=in_channel_axis, split_num=groups)
        filter_split_list = ConvUtil.split(
            filters, axis=filter_out_axis, split_num=groups
        )
        out_list = []
        name = name if name is not None else id_util.UniqueStr("Conv2d_")
        for i in range(len(in_split_list)):
            out_list.append(
                conv2d_op(
                    in_split_list[i],
                    filter_split_list[i],
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
        return conv2d_op(
            inputs,
            filters,
            padding_before,
            channel_pos,
            kernel_size_list,
            strides,
            dilations,
            groups,
            name,
        )


def conv2d_op(
    inputs,
    filters,
    padding_before,
    channel_pos,
    kernel_size_list,
    strides,
    dilations,
    groups,
    name,
):
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Conv2d_"))
        .Op("conv2d")
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
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("nn.conv3d")
def conv3d(
    input: remote_blob_util.BlobDef,
    filters: remote_blob_util.BlobDef,
    strides: Union[int, Sequence[int]],
    padding: Union[str, Tuple[IntPair, IntPair, IntPair, IntPair, IntPair]],
    data_format: str = "NCDHW",
    dilations: Optional[Union[int, Sequence[int]]] = None,
    groups: int = 1,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""3D convolution layer.

    Args:
        input (remote_blob_util.BlobDef):  A 5D input `Blob`. [batch_num, channel, depth, height, width]
        filters (remote_blob_util.BlobDef): A `Blob` with the same type as `input` and has the shape `[out_channels, in_channels//groups, filter_depth, filter_height, filter_width] for NCDHW, or [out_channels, filter_depth, filter_height, filter_width, in_channels//groups] for NDHWC`
        strides (Union[int, Sequence[int]]): An `int` or `list of ints` that has length `3`. The stride of the sliding window for each dimension of `input`.
        padding (Union[str, Tuple[IntPair, IntPair, IntPair, IntPair, IntPair]]): padding: `string` `"SAME"` or `"SAME_LOWER"` or `"SAME_UPPER"` or `"VALID" or Tuple[IntPair, IntPair, IntPair, IntPair, IntPair]` indicating the type of padding algorithm to use, or a list indicating the explicit paddings at the start and end of each dimension.
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
        remote_blob_util.BlobDef: A `Blob` with the same type as `input` and the same outer batch shape.
    
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
    if data_format.upper() == "NDHWC":  # NDHWC is not supported before cudnn 8.0
        need_transpose = 1
        data_format = "NCDHW"
    if need_transpose:
        input = flow.transpose(input, perm=[0, 4, 1, 2, 3])
        filters = flow.transpose(filters, perm=[0, 4, 1, 2, 3])

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
    else:
        if isinstance(dilations, (list, tuple)):
            assert len(dilations) == 3, ValueError(
                "dilations length must be 3 when passed as a list."
            )
        elif isinstance(dilations, int):
            dilations = [dilations, dilations, dilations]
        else:
            raise ValueError("dilations must be an int or a list.")

    if channel_pos == "channels_first":
        kernel_size_list = filters.shape[2:5]
    elif channel_pos == "channels_last":
        kernel_size_list = filters.shape[-4:-1]
    else:
        raise ValueError("invalid data_format")
    assert isinstance(kernel_size_list, tuple)
    assert isinstance(groups, int)
    assert groups > 0
    if groups > 1:
        if data_format.upper() == "NCDHW":
            assert groups <= filters.shape[0]
            assert filters.shape[0] % groups == 0
            assert groups <= input.shape[1]
            assert input.shape[1] % groups == 0
            assert filters.shape[1] == input.shape[1] // groups
        elif data_format.upper() == "NDHWC":
            raise ValueError("data_format NHWC not support groups > 1")
        else:
            raise ValueError("invalid data_format")
    inputs, pads_list = calc_conv_padding(
        input, padding, data_format.upper(), kernel_size_list, dilations, strides,
    )
    assert len(pads_list) == len(inputs.shape) - 2
    padding_before = [pad[0] for pad in pads_list]
    output = (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Conv3d_"))
        .Op("conv3d")
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
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )

    if need_transpose:
        output = flow.transpose(output, perm=[0, 2, 3, 4, 1])
    return output


@oneflow_export("nn.moments")
def moments(
    x: remote_blob_util.BlobDef,
    axes: List[int],
    keepdims: Optional[bool] = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    """This operator computes the mean and variance value of input Blob.

    Args:
        x (remote_blob_util.BlobDef): A Blob
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


@oneflow_export("nn.batch_normalization")
def batch_normalization(
    x: remote_blob_util.BlobDef,
    mean: remote_blob_util.BlobDef,
    variance: remote_blob_util.BlobDef,
    offset: Optional[remote_blob_util.BlobDef] = None,
    scale: Optional[remote_blob_util.BlobDef] = None,
    variance_epsilon: Optional[float] = 1e-5,
    axis: int = 1,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""This op does not fully align with tf.nn.batch_normalization. 
    
    The `mean`, `variable`, `offset` and `scale` are always 1D. Users need to specify `axis` to 1 for NCHW data format.

    Args:
        x (remote_blob_util.BlobDef): Input `Blob` of arbitrary dimensionality.
        mean (remote_blob_util.BlobDef): A 1D mean `Blob`.
        variance (remote_blob_util.BlobDef):   A 1D variance `Blob`.
        offset (Optional[remote_blob_util.BlobDef]): An 1D offset `Blob`, often denoted  in equations, or None. If present, will be added to the normalized `Blob`.
        scale (Optional[remote_blob_util.BlobDef]): A 1D scale `Blob`, often denoted  in equations, or None. If present, the scale is applied to the normalized `Blob`.
        variance_epsilon (float):   A small float number to avoid dividing by 0.
        axis (int, optional): 1 for '`NCHW'` data format. Defaults to 1.
        name (Optional[str], optional): This operator's name.

    Returns:
        remote_blob_util.BlobDef:  the normalized, scaled, offset `Blob`.
    
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
            # momentum is not used
            .Attr("momentum", 0.0)
        )
        return builder.Build().InferAndTryRun().RemoteBlobList()[0]
    else:
        raise NotImplementedError


@oneflow_export("nn.layer_norm")
def layer_norm(
    inputs: remote_blob_util.BlobDef,
    gamma: Optional[remote_blob_util.BlobDef] = None,
    beta: Optional[remote_blob_util.BlobDef] = None,
    begin_norm_axis: int = 1,
    begin_params_axis: int = -1,
    epsilon: float = 1e-5,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Layer Normalization.

    Args:
        inputs (remote_blob_util.BlobDef): Input `Blob`.
        gamma (Optional[remote_blob_util.BlobDef]).
        beta (Optional[remote_blob_util.BlobDef]).
        begin_norm_axis (int, optional): An integer specifies which axis to normalize at first. Defaults to 1.
        begin_params_axis (int, optional):  An integer specifies which axis params at . Defaults to -1.
        epsilon (float, optional): A small float is added to avoid division by zero. Defaults to 1e-5.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Returns:
        remote_blob_util.BlobDef: A normalized `Blob` with same shape of input.

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


@oneflow_export("nn.compat_conv2d")
def tf_conv2d(
    input: remote_blob_util.BlobDef,
    filters: remote_blob_util.BlobDef,
    strides: Union[int, Sequence[int]],
    padding: str,
    data_format: str = "NCHW",
    dilations: Optional[Union[int, Sequence[int]]] = None,
    groups: int = 1,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Computes a 2-D convolution given `input` and 4-D `filters` `Blob`.

    Args:
        input (remote_blob_util.BlobDef): A `Blob` of rank at least 4.
        filters (remote_blob_util.BlobDef): A `Blob` with the same type as `input` and has the shape `[out_channels, in_channels//groups, filter_height, filter_width] for NCHW, or [out_channels, filter_height, filter_width, in_channels//groups] for NHWC`
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
        remote_blob_util.BlobDef:  A `Blob` with the same type as `input` and the same outer batch shape.
    
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
        input, filters, strides, padding, data_format, dilations, groups, name
    )


@oneflow_export("nn.bias_add")
def bias_add(
    value: remote_blob_util.BlobDef,
    bias: remote_blob_util.BlobDef,
    data_format: Optional[str] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""This operator adds a bias to Blob. 

    Args:
        value (remote_blob_util.BlobDef):  A `Blob`.
        bias (remote_blob_util.BlobDef): A 1-D `Blob` with size matching the channel dimension of value. And has the same type as value unless value is a quantized type.
        data_format (Optional[str], optional): A string. '`N...C'` or '`NC...'`. Defaults to None.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Raises:
        ValueError: ValueError if data format is unrecognized, if value has less than two dimensions with '`N..C'`/None data_format or value has less than three dimensions with '`NC..'` data_format, if bias is a vector, or if the size of bias does not match the size of the channel dimension of value.

    Returns:
        remote_blob_util.BlobDef: A `Blob` with the same type as value.
    
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
    # TODO: name unused, fix it
    if name is None:
        name = id_util.UniqueStr("BiasAdd_")

    if data_format is None:
        bias_add_axis = 1
    else:
        if data_format.startswith("NC"):
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


@oneflow_export("nn.max_pool1d")
def max_pool1d(
    input: remote_blob_util.BlobDef,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Sequence[int]]],
    data_format: str = "NWC",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Performs the 1d-max pooling on the input.

    Args:
        input (remote_blob_util.BlobDef): A 3-D `Blob` of the format specified by data_format.
        ksize (Union[int, Sequence[int]]): An int or list of ints that has length 1 or 3. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1 or 3. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str):  '`VALID'` or '`SAME'`. The padding algorithm.
        data_format (str, optional):  An optional string from: '`NWC'`, '`NCW'`. Defaults to '`NWC'`.
        name (Optional[str], optional): This operator's name(optional).Defaults to None.

    Raises:
        NotImplementedError: TODO: fix cuDNN bugs in pooling_1d

    Returns:
        remote_blob_util.BlobDef: A `Blob` of format specified by data_format. The max pooled output `Blob`.
    """
    # TODO: fix cuDNN bugs in pooling_1d
    raise NotImplementedError


@oneflow_export("nn.avg_pool1d")
def avg_pool1d(
    input: remote_blob_util.BlobDef,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Sequence[int]]],
    data_format: str = "NCW",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Performs the average pooling on the input `Blob`.

    Args:
        input (remote_blob_util.BlobDef): A 3-D `Blob` of the format specified by data_format.
        ksize (Union[int, Sequence[int]]): An int or list of ints that has length 1 or 3. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1 or 3. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'`.
        data_format (str, optional):  '`NWC'` or '`NCW'`. Defaults to '`NWC'`.
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Raises:
        NotImplementedError: TODO: fix cuDNN bugs in pooling_1d

    Returns:
        remote_blob_util.BlobDef: A `Blob` of format specified by data_format. The max pooled output `Blob`.
    """
    # TODO: fix cuDNN bugs in pooling_1d
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
    return padding_type, ndim_pads_list


@oneflow_export("nn.max_pool2d")
def max_pool2d(
    input: remote_blob_util.BlobDef,
    ksize: Union[int, IntPair],
    strides: Union[int, IntPair],
    padding: Union[str, Tuple[IntPair, IntPair, IntPair, IntPair]],
    data_format: str = "NCHW",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r""" Performs the 2d-max pooling on the input `Blob`.

    Args:
        input (remote_blob_util.BlobDef): A 4-D `Blob` of the format specified by data_format.
        ksize (Union[int, IntPair]): An int or list of ints that has length 1, 2. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, IntPair]): An int or list of ints that has length 1, 2. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'` or '`SAME_LOWER'` or '`SAME_UPPER'` or Tuple[IntPair, IntPair, IntPair, IntPair]`. The padding algorithm.
        data_format (str, optional): '`NHWC'`, '`NCHW'` or '`NCHW_VECT_C'`. Defaults to "NCHW".
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef:  A `Blob` of format specified by data_format. The max pooled output `Blob`.
    
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
    padding_type, pads_list = calc_pool_padding(padding, get_dhw_offset(channel_pos), 2)
    assert len(pads_list) == len(input.shape) - 2
    padding_before = [pad[0] for pad in pads_list]
    padding_after = [pad[1] for pad in pads_list]
    op.Attr("padding", padding_type)
    op.Attr("padding_before", padding_before)
    op.Attr("padding_after", padding_after)
    op.Attr("ceil_mode", ceil_mode)
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.avg_pool2d")
def avg_pool2d(
    input: remote_blob_util.BlobDef,
    ksize: Union[int, IntPair],
    strides: Union[int, IntPair],
    padding: Union[str, Tuple[IntPair, IntPair, IntPair, IntPair]],
    data_format: str = "NCHW",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Performs the 2d-average pooling on the input. 

    Args:
        input (remote_blob_util.BlobDef): A 4-D `Blob` of shape [batch, height, width, channels].
        ksize (Union[int, IntPair]):  An int or list of ints that has length 1, 2. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, IntPair]): An int or list of ints that has length 1, 2. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'` or '`SAME_LOWER'` or '`SAME_UPPER'` or Tuple[IntPair, IntPair, IntPair, IntPair]. The padding algorithm.
        data_format (str, optional): '`NHWC'` or '`NCHW'`. Defaults to "NCHW".
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef:  A `Blob` with the same type as '`value'`. The average pooled output `Blob`.
    
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
    padding_type, pads_list = calc_pool_padding(padding, get_dhw_offset(channel_pos), 2)
    assert len(pads_list) == len(input.shape) - 2
    padding_before = [pad[0] for pad in pads_list]
    padding_after = [pad[1] for pad in pads_list]
    op.Attr("padding", padding_type)
    op.Attr("padding_before", padding_before)
    op.Attr("padding_after", padding_after)
    op.Attr("ceil_mode", ceil_mode)
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.max_pool3d")
def max_pool3d(
    input: remote_blob_util.BlobDef,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Sequence[int]]],
    data_format: str = "NCDHW",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Performs the 3d-max pooling on the input.

    Args:
        input (remote_blob_util.BlobDef):  A 5-D `Blob` of the format specified by data_format.
        ksize (Union[int, Sequence[int]]):  An int or list of ints that has length 1, 3 or 5. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1, 3 or 5. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'` or '`SAME_LOWER'` or '`SAME_UPPER'` or '`Sequence[Sequence[int]]'`.
        data_format (str, optional):   "NDHWC" or "NCDHW". Defaults to "NCDHW".
        name (Optional[str], optional): This operator's name(optional).

    Returns:
        remote_blob_util.BlobDef: A `Blob` of format specified by data_format. The max pooled output `Blob`.
    
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
    padding_type, pads_list = calc_pool_padding(padding, get_dhw_offset(channel_pos), 3)
    assert len(pads_list) == len(input.shape) - 2
    padding_before = [pad[0] for pad in pads_list]
    padding_after = [pad[1] for pad in pads_list]
    op.Attr("padding", padding_type)
    op.Attr("padding_before", padding_before)
    op.Attr("padding_after", padding_after)
    op.Attr("ceil_mode", ceil_mode)
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.avg_pool3d")
def avg_pool3d(
    input: remote_blob_util.BlobDef,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Sequence[int]]],
    data_format: str = "NCDHW",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Performs the 3d-average pooling on the input. 

    Args:
        input (remote_blob_util.BlobDef): A 5-D `Blob` of shape [batch, height, width, channels].
        ksize (Union[int, Sequence[int]]): An int or list of ints that has length 1, 3 or 5. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1, 3 or 5. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'` or '`SAME_LOWER'` or '`SAME_UPPER or Sequence[Sequence[int]]'`.
        data_format (str, optional):  '`NDHWC'` or '`NCDHW'`. Defaults to "NCDHW".
        name (Optional[str], optional):  This operator's name(optional).Defaults to None.

    Returns:
        remote_blob_util.BlobDef: A `Blob` with the same type as value. The average pooled output `Blob`.
    
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
    padding_type, pads_list = calc_pool_padding(padding, get_dhw_offset(channel_pos), 3)
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
    assert axis >= 1
    assert axis < dim_num

    need_transpose = False
    permute = [i for i in range(dim_num)]
    if axis > 0 and axis != dim_num - 1:
        need_transpose = True
        permute[axis] = permute[-1]
        permute[-1] = axis
    return need_transpose, permute


@oneflow_export("nn.softmax")
def softmax(
    logits: remote_blob_util.BlobDef,
    axis: Optional[int] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Computes softmax activations. 

    For each element, we apply: 

    .. math::
        S_i = \frac{e^i}{\sum_1^j e^j }

    Args:
        logits (remote_blob_util.BlobDef): A non-empty `Blob`.
        axis (Optional[int], optional): The dimension softmax would be performed on. Defaults to None.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef:  A `Blob` has the same type and shape as logits.

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

    need_transpose, permute = _softmax_need_transpose(logits, axis)
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


@oneflow_export("nn.softmax_grad")
def softmax_grad(
    y: remote_blob_util.BlobDef,
    dy: remote_blob_util.BlobDef,
    axis: Optional[int] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Computes gradient of softmax activations.

    Args:
        y (remote_blob_util.BlobDef):  A `Blob` representing the softmax of x.
        dy (remote_blob_util.BlobDef):  gradient of y.
        axis (Optional[int], optional):  The dimension softmax would be performed on. Defaults to None.
        name (Optional[str], optional):  This operator's name(optional).

    Returns:
        remote_blob_util.BlobDef:  A `Blob` representing the gradient of x.
    """
    if axis is None:
        axis = -1

    need_transpose, permute = _softmax_need_transpose(y, axis)
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


@oneflow_export("nn.sparse_cross_entropy")
def sparse_cross_entropy(
    labels: remote_blob_util.BlobDef,
    prediction: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Computes sparse cross entropy

    Args:
        labels (remote_blob_util.BlobDef): A `Blob` of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels and result). Each entry in labels must be an index in [0, num_classes).
        prediction (remote_blob_util.BlobDef): A `Blob` with the rank that is equal to the rank of the labels plus one.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef: A `Blob` of the same shape as labels.
    
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

    if prediction.distribute is distribute_util.split(len(prediction.shape) - 1):
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


@oneflow_export("nn.softmax_cross_entropy_with_logits")
def softmax_cross_entropy_with_logits(
    labels: remote_blob_util.BlobDef,
    logits: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Computes softmax cross entropy between logits and labels. 

    Args:
        labels (remote_blob_util.BlobDef): Each vector along the class dimension should hold a valid probability distribution.
        logits (remote_blob_util.BlobDef): Per-label activations, typically a linear output. logits has same shape and dtype as labels.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef: A `Blob` that contains the softmax cross entropy loss. Its type is the same as logits and its shape is the same as labels except that it does not have the last dimension of labels.
    
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

    prob, out = (
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


@oneflow_export("nn.sparse_softmax_cross_entropy_with_logits")
def sparse_softmax_cross_entropy_with_logits(
    labels: remote_blob_util.BlobDef,
    logits: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Computes sparse softmax cross entropy between logits and labels. 

    Args:
        labels (remote_blob_util.BlobDef): `Blob` of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels and result). Each entry in labels must be an index in [0, num_classes).
        logits (remote_blob_util.BlobDef): Unscaled log probabilities of shape [d_0, d_1, ..., d_{r-1},num_classes].
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Raises:
        ValueError: If logits are scalars (need to have rank >= 1) or if the rank of the labels is not equal to the rank of the logits minus one.

    Returns:
        remote_blob_util.BlobDef:  A `Blob` of the same shape as labels and of the same type as logits with the softmax cross entropy loss.
    
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

    if logits.distribute is distribute_util.split(len(logits.shape) - 1):
        prob, out = (
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
        prob, out = (
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


@oneflow_export("nn.sigmoid_cross_entropy_with_logits")
def sigmoid_cross_entropy_with_logits(
    labels: remote_blob_util.BlobDef,
    logits: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Computes sigmoid cross entropy given logits. 

    Args:
        labels (remote_blob_util.BlobDef): A `Blob` of the same type and shape as logits.
        logits (remote_blob_util.BlobDef): A `Blob` of type float.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef:   A `Blob` of the same shape as logits with the componentwise logistic losses.

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
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("SigmoidCrossEntropy_"),
    )
    op_conf.sigmoid_cross_entropy_conf.prediction = logits.unique_name
    op_conf.sigmoid_cross_entropy_conf.label = labels.unique_name
    op_conf.sigmoid_cross_entropy_conf.loss = "loss"
    op_conf.sigmoid_cross_entropy_conf.label_type = labels.dtype.oneflow_proto_dtype
    interpret_util.Forward(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "loss"
    return remote_blob_util.RemoteBlob(lbi)


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


@oneflow_export("nn.random_mask_like")
def random_mask_like(
    like: remote_blob_util.BlobDef,
    rate: float,
    seed: Optional[int] = None,
    noise_shape: Optional[Sequence] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Random mask `Blob` with same shape as '`like'`.

    Args:
        like (remote_blob_util.BlobDef): A `Blob`.
        rate (float): A float value for the probability that each element is dropped.
        seed (Optional[int], optional): Optional, int value. Defaults to None.
        noise_shape (Optional[Sequence], optional): Optional, A 1-D `Blob`, representing the shape for randomly generated keep/drop flags. Defaults to None.
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef: A random mask `Blob` of the same shape of `like`.

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
    assert rate is not None and rate >= 0.0 and rate < 1.0
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
            name, lambda: RandomMaskLike(rate=rate, seed=seed, name=name,),
        )
        return module(like)


class RandomMaskLike(module_util.Module):
    def __init__(
        self, rate: float, seed: Optional[int] = None, name: str = None,
    ):
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

    def forward(self, like: remote_blob_util.BlobDef):
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


@oneflow_export("nn.dropout")
def dropout(
    x: remote_blob_util.BlobDef,
    rate: float,
    noise_shape: Optional[remote_blob_util.BlobDef] = None,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""For preventing overfitting, randomly set elements to zero. 

    Args:
        x (remote_blob_util.BlobDef): A floating point `Blob`.
        rate (float): A scalar `Blob` with the same type as x. The probability that each element is dropped.
        noise_shape (Optional[remote_blob_util.BlobDef], optional):  optional: A 1-D `Blob`, representing the shape for randomly generated keep/drop flags. Defaults to None.Defaults to None.
        seed (Optional[int], optional):  Optional int value. Defaults to None.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef:   A `Blob` of the same shape of x.

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
    assert rate is not None and rate >= 0.0 and rate < 1.0
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


@oneflow_export("nn.conv2d_transpose")
def deconv2d(
    value: Optional[remote_blob_util.BlobDef] = None,
    filter: Optional[remote_blob_util.BlobDef] = None,
    output_shape: Tuple[int, int, int, int] = None,
    strides: Optional[Union[int, Sequence[int]]] = None,
    padding: str = "VALID",
    data_format: str = "NCHW",
    name: Optional[str] = None,
    input: Optional[remote_blob_util.BlobDef] = None,
    filters: Optional[remote_blob_util.BlobDef] = None,
    dilations: Optional[Union[int, Sequence[int]]] = None,
) -> remote_blob_util.BlobDef:
    r"""2d transposed convolution.

    Args:
        value (Optional[remote_blob_util.BlobDef], optional):   4-d `Blob`. Defaults to None.
        filter (Optional[remote_blob_util.BlobDef], optional): Filter of transposed convolution, usually a variable. Defaults to None.
        output_shape (Tuple[int, int, int, int]): A 1-D `Blob` representing the output shape of the deconvolution op. Defaults to None.
        strides (Optional[Union[int, Sequence[int]]], optional): `int` or `int list`. Defaults to None.
        padding (str, optional):  `'VALID'` or `'SAME'`. Defaults to "VALID".
        data_format (str, optional): `'NHWC'` or `'NCHW'`. Defaults to "NCHW".
        name (Optional[str], optional): This operator's name(optional). Defaults to None.
        input (Optional[remote_blob_util.BlobDef], optional): Alias for value. Defaults to None.
        filters (Optional[remote_blob_util.BlobDef], optional): Alias for filter. Defaults to None.
        dilations (Optional[Union[int, Sequence[int]]], optional): The dilation factor for each dimension of input. Defaults to None.

    Raises:
        ValueError: shapes of `filter` and `input` must match.
        ValueError: dilations must be an int or a list.
        ValueError: data_format must be "NHWC" or "NCHW".
        ValueError: padding must be "SAME" or "VALID".

    Returns:
        remote_blob_util.BlobDef: A `Blob` with the same type as `value`.

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

    # dilations
    if dilations is None:
        dilations = [1, 1]
    else:
        if isinstance(dilations, (list, tuple)):
            assert len(dilations) == 2, ValueError(
                "dilations length must be 2 when passed as a list."
            )
        elif isinstance(dilations, int):
            dilations = [dilations, dilations]
        else:
            raise ValueError("dilations must be an int or a list.")

    # data format
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

    # strides
    if isinstance(strides, (list, tuple)):
        assert len(strides) == NDims, ValueError(
            "strides length must be 2 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")

    # output_padding and padding_needed
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
    # add pad op if needs odd padding
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


@oneflow_export("nn.torch_conv2d_transpose")
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

    # dilations
    if dilations is None:
        dilations = [1, 1]
    else:
        if isinstance(dilations, (list, tuple)):
            assert len(dilations) == 2, ValueError(
                "dilations length must be 2 when passed as a list."
            )
        elif isinstance(dilations, int):
            dilations = [dilations, dilations]
        else:
            raise ValueError("dilations must be an int or a list.")

    # data format
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

    # strides
    if isinstance(strides, (list, tuple)):
        assert len(strides) == NDims, ValueError(
            "strides length must be 2 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")

    # output_padding and padding_needed
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


@oneflow_export("nn.leaky_relu")
def leaky_relu(
    x: remote_blob_util.BlobDef, alpha: float = 0.2, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Leaky ReLU activation. 

    .. math::
        out = max(x, alpha*x)

    Args:
        x (remote_blob_util.BlobDef):  A `Blob` representing preactivation values.
        alpha (float, optional): Slope of the activation function at x < 0 with float type. Default value is 0.2.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef: The activation `Blob`.

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


@oneflow_export("nn.L1Loss")
def l1_loss(
    input: remote_blob_util.BlobDef,
    target: remote_blob_util.BlobDef,
    reduction: str = "mean",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""This operator computes the L1 Loss between each element in `input` and `target`. 

    The equation is: 

    if reduction = "none": 
    
    .. math:: 

        output = |Target - Input|

    if reduction = "mean": 
    
    .. math:: 

        output = \frac{1}{n}\sum_{i=1}^n|Target_i - Input_i|

    if reduction = "sum": 
    
    .. math:: 
    
        output = \sum_{i=1}^n|Target_i - Input_i|

    Args:
        input (remote_blob_util.BlobDef): The input Blob.  
        target (remote_blob_util.BlobDef): The target value. 
        reduction (str): The reduce type, it can be one of "none", "mean", "sum". Defaults to "mean".
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        remote_blob_util.BlobDef: The result Blob. 

    For example: 

    Example 1: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        @flow.global_function()
        def l1_job(x: tp.Numpy.Placeholder(shape=(3, 3)),
                y: tp.Numpy.Placeholder(shape=(3, 3))) -> tp.Numpy:
            out = flow.nn.L1Loss(x, y, reduction="mean", name="l1")

            return out


        input = np.array([[1, 1, 1], [2, 2, 2], [7, 7, 7]]).astype(np.float32)
        target = np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]]).astype(np.float32)

        out = l1_job(input, target)

        # output [2.6666667]

    Example 2: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        @flow.global_function()
        def l1_job(x: tp.Numpy.Placeholder(shape=(3, 3)),
                y: tp.Numpy.Placeholder(shape=(3, 3))) -> tp.Numpy:
            out = flow.nn.L1Loss(x, y, reduction="sum", name="l1")

            return out


        input = np.array([[1, 1, 1], [2, 2, 2], [7, 7, 7]]).astype(np.float32)
        target = np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]]).astype(np.float32)

        out = l1_job(input, target)

        # output [24.]

    """
    assert (
        input.shape == target.shape
    ), "The Input shape must be the same as Target shape"

    assert reduction in [
        "none",
        "mean",
        "sum",
    ], "{} is not a valid value for reduction, The reduction must be the one of `none`, `mean`, `sum`. ".format(
        reduction
    )

    if name is None:
        name = "L1Loss"

    l1_value = flow.math.abs(
        flow.math.subtract(target, input, name=name + "_sub"), name=name + "_abs"
    )

    if reduction == "mean":
        return flow.math.reduce_mean(l1_value, name=name + "_reduce_mean")
    elif reduction == "sum":
        return flow.math.reduce_sum(l1_value, name=name + "_reduce_sum")
    else:
        # Do no reduction
        return l1_value


@oneflow_export("nn.BCELoss")
def bce_loss(
    input: remote_blob_util.BlobDef,
    target: remote_blob_util.BlobDef,
    weight: remote_blob_util = None,
    reduction: str = "mean",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""This operator computes the binary cross entropy loss. 
    
    The equation is: 

    if reduction = "none": 

    .. math:: 

        out = -(Target_i*ln(Input_i) + (1-Target_i)*ln(1-Input_i))
    
    if reduction = "mean": 

    .. math:: 
        
        out = -\frac{1}{n}\sum_{i=1}^n(Target_i*ln(Input_i) + (1-Target_i)*ln(1-Input_i))
    
    if reduction = "sum": 
    
    .. math:: 
        
        out = -\sum_{i=1}^n(Target_i*ln(Input_i) + (1-Target_i)*ln(1-Input_i))
    Args:
        input (remote_blob_util.BlobDef): The input Blob. 
        target (remote_blob_util.BlobDef): The target value. 
        weight (remote_blob_util, optional): The manual rescaling weight to the loss. Default to None, whose corresponding weight value is 1.
        reduction (str, optional): The reduce type, it can be one of "none", "mean", "sum". Defaults to "mean".
        name (Optional[str], optional): The name for the operation. Defaults to None.
    Returns:
        remote_blob_util.BlobDef: The result Blob. 
    """
    assert (
        input.shape == target.shape
    ), "The Input shape must be the same as Target shape"
    assert reduction in [
        "none",
        "mean",
        "sum",
    ], "{} is not a valid value for reduction, The reduction must be the one of `none`, `mean`, `sum`. ".format(
        reduction
    )

    if name is None:
        name = "BCELoss"

    _sigmiod_value = flow.math.sigmoid(input, name=name + "_sigmoided_input")
    _cross_entropy_loss = flow.math.negative(
        target * flow.math.log(_sigmiod_value)
        + (1 - target) * flow.math.log(1 - _sigmiod_value)
    )

    if weight is not None:
        assert (
            weight.shape == input.shape
        ), "The weight shape must be the same as Input shape"
        _weighted_loss = weight * _cross_entropy_loss
    else:
        _weighted_loss = _cross_entropy_loss
    if reduction == "mean":
        return flow.math.reduce_mean(_weighted_loss, name=name + "_reduce_mean")
    elif reduction == "sum":
        return flow.math.reduce_sum(_weighted_loss, name=name + "_reduce_sum")
    else:
        # Do no reduction
        return _weighted_loss
