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
import random
from typing import Union, Optional, Sequence
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


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
                    padding.upper(),
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


@oneflow_export("nn.conv2d")
def conv2d(
    input: remote_blob_util.BlobDef,
    filters: remote_blob_util.BlobDef,
    strides: Union[int, Sequence[int]],
    padding: Union[str, Sequence[Sequence[int]]],
    data_format: str = "NHWC",
    dilations: Optional[Union[int, Sequence[int]]] = None,
    groups: int = 1,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""2d convolution 

    Analogous to `tf.nn.conv2d <https://www.tensorflow.org/api_docs/python/tf/nn/conv2d>`_

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

    if channel_pos == "channels_first":
        kernel_size_list = filters.shape[2:4]
    elif channel_pos == "channels_last":
        kernel_size_list = filters.shape[-3:-1]
    else:
        raise ValueError("invalid data_format")
    assert isinstance(kernel_size_list, tuple)
    assert isinstance(groups, int)
    assert groups > 0
    if groups > 1:
        if data_format.upper() == "NCHW":
            assert groups <= filters.shape[0]
            assert filters.shape[0] % groups == 0
            assert groups <= input.shape[1]
            assert input.shape[1] % groups == 0
            assert filters.shape[1] == input.shape[1] // groups
        elif data_format.upper() == "NHWC":
            raise ValueError("data_format NHWC not support groups > 1")
        else:
            raise ValueError("invalid data_format")
    inputs, pads_list = calc_conv_padding(
        input, padding, data_format.upper(), kernel_size_list, dilations, strides,
    )
    assert len(pads_list) == len(inputs.shape) - 2
    padding_before = [pad[0] for pad in pads_list]

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


@oneflow_export("nn.batch_normalization")
def batch_normalization(
    x: remote_blob_util.BlobDef,
    mean: remote_blob_util.BlobDef,
    variance: remote_blob_util.BlobDef,
    offset: remote_blob_util.BlobDef,
    scale: remote_blob_util.BlobDef,
    variance_epsilon: float,
    axis: int = -1,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""
    This op does not fully align with tf.nn.batch_normalization. mean, variable, offset and scale
    are always 1D. Users need to specify "axis" to 1 for NCHW data format.

    """

    assert axis >= -len(x.shape) and axis < len(x.shape)
    if axis < 0:
        axis += len(x.shape)

    if name is None:
        name = id_util.UniqueStr("BatchNorm_")

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


@oneflow_export("nn.compat_conv2d")
def tf_conv2d(
    input: remote_blob_util.BlobDef,
    filters: remote_blob_util.BlobDef,
    strides: Union[int, Sequence[int]],
    padding: str,
    data_format: str = "NHWC",
    dilations: Optional[Union[int, Sequence[int]]] = None,
    groups: int = 1,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    assert len(input.shape) == 4
    assert len(filters.shape) == 4
    NDims = 2
    if isinstance(strides, (list, tuple)):
        assert len(strides) == 2, ValueError(
            "strides length must be 2 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")

    if padding.upper() != "SAME" and padding.upper() != "VALID":
        raise ValueError('padding must be "SAME" or "VALID".')

    if data_format.upper() != "NCHW" and data_format.upper() != "NHWC":
        raise ValueError('data_format must be "NHWC" or "NCHW".')

    channel_pos = "channels_first" if data_format.startswith("NC") else "channels_last"

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

    if channel_pos == "channels_first":
        input_size = input.shape[2:4]
        kernel_size_list = filters.shape[2:4]
    elif channel_pos == "channels_last":
        input_size = input.shape[-3:-1]
        kernel_size_list = filters.shape[-3:-1]
    else:
        raise ValueError("invalid data_format")
    # add pad op if needs odd padding
    if padding.upper() == "SAME":
        padding_left = [0] * NDims
        padding_right = [0] * NDims
        for i in range(NDims):
            effective_filter_size = (kernel_size_list[i] - 1) * dilations[i] + 1
            tmp_output_size = (input_size[i] + strides[i] - 1) // strides[i]
            padding_needed = max(
                0,
                (tmp_output_size - 1) * strides[i]
                + effective_filter_size
                - input_size[i],
            )
            padding_left[i] = padding_needed // 2
            padding_right[i] = padding_needed - padding_needed // 2
        if padding_left != padding_right:
            assert data_format.upper() == "NCHW"
            input = flow.pad(
                input,
                [
                    (0, 0),
                    (0, 0),
                    (padding_left[0], padding_right[0]),
                    (padding_left[1], padding_right[1]),
                ],
                name=name + "_pad" if name is not None else None,
            )
            padding = "VALID"
    assert isinstance(kernel_size_list, (list, tuple))
    assert isinstance(groups, int)
    assert groups > 0
    if groups > 1:
        if data_format.upper() == "NCHW":
            assert groups <= filters.shape[0]
            assert filters.shape[0] % groups == 0
            assert groups <= input.shape[1]
            assert input.shape[1] % groups == 0
            assert filters.shape[1] == input.shape[1] // groups
        elif data_format.upper() == "NHWC":
            raise ValueError("data_format NHWC not support groups > 1")
        else:
            raise ValueError("invalid data_format")
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Conv2d_"))
        .Op("conv2d")
        .Input("in", [input])
        .Input("weight", [filters])
        .Output("out")
        .Attr("filters", filters.shape[0])
        .Attr("padding", padding.lower())
        .Attr("data_format", channel_pos)
        .Attr("kernel_size", kernel_size_list)
        .Attr("strides", strides)
        .Attr("dilation_rate", dilations)
        .Attr("groups", groups)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("nn.bias_add")
def bias_add(
    value: remote_blob_util.BlobDef,
    bias: remote_blob_util.BlobDef,
    data_format: Optional[str] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""
    Analogous to `tf.nn.bias_add <https://www.tensorflow.org/api_docs/python/tf/nn/bias_add>`_

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
    padding: str,
    data_format: str = "NWC",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    # TODO: fix cuDNN bugs in pooling_1d
    raise NotImplementedError


@oneflow_export("nn.avg_pool1d")
def avg_pool1d(
    input: remote_blob_util.BlobDef,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    data_format: str = "NWC",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
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
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    data_format: str = "NHWC",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""
    Analogous to `tf.nn.max_pool2d <https://www.tensorflow.org/api_docs/python/tf/nn/max_pool2d>`_

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
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    data_format: str = "NHWC",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""
    Analogous to `tf.nn.avg_pool2d <https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool2d>`_

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
    padding: str,
    data_format: str = "NDHWC",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""
    Analogous to `tf.nn.max_pool3d <https://www.tensorflow.org/api_docs/python/tf/nn/max_pool3d>`_

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
    padding: str,
    data_format: str = "NDHWC",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""
    Analogous to `tf.nn.avg_pool3d <https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool3d>`_

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
    r"""
    Analogous to `tf.nn.softmax <https://www.tensorflow.org/api_docs/python/tf/nn/softmax>`_

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
    assert labels is not None
    assert prediction is not None

    if len(labels.shape) == len(prediction.shape):
        assert labels.shape[-1] == 1
        labels = flow.squeeze(labels, axis=[-1])
    else:
        assert len(labels.shape) == len(prediction.shape) - 1

    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("SparseCrossEntropy_")
        )
        .Op("sparse_cross_entropy")
        .Input("prediction", [prediction])
        .Input("label", [labels])
        .Output("out")
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
    r"""
    Analogous to `tf.nn.softmax_cross_entropy_with_logits <https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits>`_

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
    r"""
    Analogous to `tf.nn.sparse_softmax_cross_entropy_with_logits <https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits>`_

    """
    assert labels is not None
    assert logits is not None

    if len(labels.shape) == len(logits.shape):
        assert labels.shape[-1] == 1
        labels = flow.squeeze(labels, axis=[-1])
    else:
        assert len(labels.shape) == len(logits.shape) - 1

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
    r"""
    Analogous to `tf.nn.sigmoid_cross_entropy_with_logits <https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits>`_

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
    assert rate is not None and rate >= 0.0 and rate < 1.0
    mask_op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("RandomMaskLike_")
        )
        .Op("random_mask_like")
        .Input("like", [like])
        .Output("out")
        .Attr("rate", float(rate))
    )
    if seed is not None:
        mask_op.Attr("seed", seed)
    else:
        mask_op.Attr("seed", random.randint(-(2 ** 63) + 1, 2 ** 63 - 1))

    if noise_shape is not None:
        assert 0, "noise_shape will be supported later."
        assert isinstance(noise_shape, (list, tuple))
    return mask_op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.dropout")
def dropout(
    x: remote_blob_util.BlobDef,
    rate: float,
    noise_shape: Optional[remote_blob_util.BlobDef] = None,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""
    Analogous to `tf.nn.dropout <https://www.tensorflow.org/api_docs/python/tf/nn/dropout>`_

    """
    assert rate is not None and rate >= 0.0 and rate < 1.0
    if not flow.current_global_function_desc().IsTrainable() or rate == 0.0:
        return x
    mask = random_mask_like(x, rate, seed, noise_shape)
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Dropout_")
        )
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
    output_shape: Optional[remote_blob_util.BlobDef] = None,
    strides: Optional[Union[int, Sequence[int]]] = None,
    padding: str = "VALID",
    data_format: str = "NHWC",
    name: Optional[str] = None,
    input: Optional[remote_blob_util.BlobDef] = None,
    filters: Optional[remote_blob_util.BlobDef] = None,
    dilations: Optional[Union[int, Sequence[int]]] = None,
) -> remote_blob_util.BlobDef:
    r"""2d transposed convolution

    Args:
    value: 4-d `Blob`
    filter: filter of transposed convolution, usually a variable
    output_shape: A 1-D Tensor representing the output shape of the deconvolution op
    strides: `int` or `int list`
    padding: `'VALID'` or `'SAME'`
    data_format: `'NHWC'` or `'NCHW'`
    name: This operator's name
    input: Alias for value
    filters: Alias for filter
    dilations: The dilation factor for each dimension of input.
    Returns:
        A `Blob` with the same type as `value`.

    Raises:
        ValueError: shapes of `filter` and `input` must match.
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
        output_shape = output_shape[2:4]
        channels = filters.shape[1]
    elif data_format.upper() == "NHWC":
        input_shape = input.shape[1:3]
        kernel_size = filters.shape[-3:-1]
        output_shape = output_shape[1:3]
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

    # check padding needed
    if padding.upper() == "VALID":
        for i in range(NDims):
            effective_filter_size = (kernel_size[i] - 1) * dilations[i] + 1
            assert (output_shape[i] + strides[i] - effective_filter_size) // strides[
                i
            ] == input_shape[i]
    elif padding.upper() == "SAME":
        padding_left = [0] * NDims
        padding_right = [0] * NDims
        for i in range(NDims):
            assert (output_shape[i] + strides[i] - 1) // strides[i] == input_shape[i]
            effective_filter_size = (kernel_size[i] - 1) * dilations[i] + 1
            padding_needed = max(
                0,
                (input_shape[i] - 1) * strides[i]
                + effective_filter_size
                - output_shape[i],
            )
            padding_left[i] = padding_needed // 2
            padding_right[i] = padding_needed - padding_needed // 2
    else:
        raise ValueError('padding must be "SAME" or "VALID".')
    # add pad op if needs odd padding
    if padding.upper() == "SAME" and padding_left != padding_right:
        assert data_format.upper() == "NCHW"
        padded_output_shape = [0] * NDims
        for i in range(NDims):
            padded_output_shape[i] = (
                output_shape[i] + padding_left[i] + padding_right[i]
            )
        input = (
            flow.user_op_builder(
                name if name is not None else id_util.UniqueStr("Conv2d_")
            )
            .Op("deconv2d")
            .Input("in", [input])
            .Input("weight", [filters])
            .Output("out")
            .Attr("filters", channels)
            .Attr("padding", "valid")
            .Attr("data_format", channel_pos)
            .Attr("kernel_size", kernel_size)
            .Attr("strides", strides)
            .Attr("dilation_rate", dilations)
            .Attr("output_shape", padded_output_shape)
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

    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Conv2d_"))
        .Op("deconv2d")
        .Input("in", [input])
        .Input("weight", [filters])
        .Output("out")
        .Attr("filters", channels)
        .Attr("padding", padding.lower())
        .Attr("data_format", channel_pos)
        .Attr("kernel_size", kernel_size)
        .Attr("strides", strides)
        .Attr("dilation_rate", dilations)
        .Attr("output_shape", output_shape)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("nn.leaky_relu")
def leaky_relu(
    x: remote_blob_util.BlobDef, alpha: float = 0.2, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
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
