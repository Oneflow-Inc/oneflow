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
from typing import Union, Optional, Sequence
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.module as module_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("nn.conv2d")
def conv2d(
    input: remote_blob_util.BlobDef,
    filters: remote_blob_util.BlobDef,
    strides: Union[int, Sequence[int]],
    padding: str,
    data_format: str = "NHWC",
    dilations: Optional[Union[int, Sequence[int]]] = None,
    groups: int = 1,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Analogous to `tf.nn.conv2d <https://www.tensorflow.org/api_docs/python/tf/nn/conv2d>`_

    Args:
        input (remote_blob_util.BlobDef): A `Blob` of rank at least 4.[batch_num, height, width, channel] 
        filters (remote_blob_util.BlobDef): A `Blob` with the same type as `input` and has the shape `[filter_height, filter_width, in_channels, out_channels]`
        strides (Union[int, Sequence[int]]): An int or list of `ints` that has length `1`, `2` or `4`. The stride of the sliding window for each dimension of `input`. 
        padding (str): padding: `string` `"SAME"` or `"VALID"` indicating the type of padding algorithm to use, or a list indicating the explicit paddings at the start and end of each dimension. 
        data_format (str, optional): `"NHWC" or "NCHW"`. Defaults to `"NHWC"`.
        dilations (Optional[Union[int, Sequence[int]]], optional):  The dilation factor for each dimension of`input`. Defaults to None.
        groups (int, optional): int value greater than 0. Defaults to 1.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Raises:
        ValueError: strides must be an int or a list.
        ValueError: padding must be "SAME" or "VALID".
        ValueError: data_format must be "NHWC" or "NCHW".
        ValueError: dilations must be an int or a list.
        ValueError: invalid data_format.
        ValueError: data_format NHWC not support groups > 1
        ValueError: invalid data_format.

    Returns:
        remote_blob_util.BlobDef: A `Blob` with the same type as `input` and the same outer batch shape.
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
    r"""This op does not fully align with tf.nn.batch_normalization. mean, variable, offset and scale
          are always 1D. Users need to specify "axis" to 1 for NCHW data format.

    Args:
        x (remote_blob_util.BlobDef): Input `Blob` of arbitrary dimensionality.
        mean (remote_blob_util.BlobDef): A 1D mean `Blob`.
        variance (remote_blob_util.BlobDef):   A 1D variance `Blob`.
        offset (remote_blob_util.BlobDef): An 1D offset `Blob`, often denoted  in equations, or None. If present, will be added to the normalized `Blob`.
        scale (remote_blob_util.BlobDef): A 1D scale `Blob`, often denoted  in equations, or None. If present, the scale is applied to the normalized `Blob`.
        variance_epsilon (float):   A small float number to avoid dividing by 0.
        axis (int, optional): 1 for '`NCHW'` data format. Defaults to -1.
        name (Optional[str], optional): This operator's name.

    Returns:
        remote_blob_util.BlobDef:  the normalized, scaled, offset `Blob`.
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
    r"""Computes a 2-D convolution given `input` and 4-D `filters` `Blob`.

    Args:
        input (remote_blob_util.BlobDef): A `Blob` of rank at least 4. 
        filters (remote_blob_util.BlobDef): A `Blob` with the same type as `input` and has the shape `[filter_height, filter_width, in_channels, out_channels]`
        strides (Union[int, Sequence[int]]): An int or list of `ints` that has length `1`, `2` or `4`. The stride of the sliding window for each dimension of `input`. 
        padding (str): `"SAME"` or `"VALID"` indicating the type of padding algorithm to use, or a list indicating the explicit paddings at the start and end of each dimension. 
        data_format (str, optional): `"NHWC"` or `"NCHW"`. Defaults to `"NHWC"`.
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
    """
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
    r"""Analogous to `tf.nn.bias_add <https://www.tensorflow.org/api_docs/python/tf/nn/bias_add>`_

    Args:
        value (remote_blob_util.BlobDef):  A `Blob`.
        bias (remote_blob_util.BlobDef): A 1-D `Blob` with size matching the channel dimension of value. And has the same type as value unless value is a quantized type.
        data_format (Optional[str], optional): A string. '`N...C'` or '`NC...'`. Defaults to None.
        name (Optional[str], optional): This operator's name. Defaults to None.

    Raises:
        ValueError: ValueError if data format is unrecognized, if value has less than two dimensions with '`N..C'`/None data_format or value has less than three dimensions with '`NC..'` data_format, if bias is a vector, 
        or if the size of bias does not match the size of the channel dimension of value.

    Returns:
        remote_blob_util.BlobDef: A `Blob` with the same type as value.
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
    r"""Performs the max pooling on the input.

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
    padding: str,
    data_format: str = "NWC",
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


@oneflow_export("nn.max_pool2d")
def max_pool2d(
    input: remote_blob_util.BlobDef,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    data_format: str = "NHWC",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r""" Performs the max pooling on the input `Blob`.Analogous to `tf.nn.max_pool2d <https://www.tensorflow.org/api_docs/python/tf/nn/max_pool2d>`_

    Args:
        input (remote_blob_util.BlobDef): A 4-D `Blob` of the format specified by data_format.
        ksize (Union[int, Sequence[int]]): An int or list of ints that has length 1, 2 or 4. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1, 2 or 4. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'`. The padding algorithm. 
        data_format (str, optional): '`NHWC'`, '`NCHW'` or '`NCHW_VECT_C'`. Defaults to "NHWC".
        name (Optional[str], optional): This operator's name(optional).. Defaults to None.

    Returns:
        remote_blob_util.BlobDef:  A `Blob` of format specified by data_format. The max pooled output `Blob`.
    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("MaxPool2D_")
        )
        .Op("max_pool_2d")
        .Input("x", [input])
        .Output("y")
    )
    assert padding in ["VALID", "SAME"]
    op.Attr("padding", padding.lower())
    assert data_format in ["NHWC", "NCHW", "NCHW_VECT_C"]
    data_format = "channels_last" if data_format == "NHWC" else "channels_first"
    op.Attr("data_format", data_format)
    pool_size = _GetSequence(ksize, 2, "ksize")
    op.Attr("pool_size", pool_size)
    strides = _GetSequence(strides, 2, "strides")
    op.Attr("strides", strides)
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.avg_pool2d")
def avg_pool2d(
    input: remote_blob_util.BlobDef,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    data_format: str = "NHWC",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Performs the average pooling on the input. Analogous to `tf.nn.avg_pool2d <https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool2d>`_

    Args:
        input (remote_blob_util.BlobDef): A 4-D `Blob` of shape [batch, height, width, channels].
        ksize (Union[int, Sequence[int]]):  An int or list of ints that has length 1, 2 or 4. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1, 2 or 4. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'`. The padding algorithm.
        data_format (str, optional): '`NHWC'` or '`NCHW'`. Defaults to "NHWC".
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef:  A `Blob` with the same type as '`value'`. The average pooled output `Blob`.
    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("AvgPool2D_")
        )
        .Op("avg_pool_2d")
        .Input("x", [input])
        .Output("y")
    )
    assert padding in ["VALID", "SAME"]
    op.Attr("padding", padding.lower())
    assert data_format in ["NHWC", "NCHW", "NCHW_VECT_C"]
    data_format = "channels_last" if data_format == "NHWC" else "channels_first"
    op.Attr("data_format", data_format)
    pool_size = _GetSequence(ksize, 2, "ksize")
    op.Attr("pool_size", pool_size)
    strides = _GetSequence(strides, 2, "strides")
    op.Attr("strides", strides)
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.max_pool3d")
def max_pool3d(
    input: remote_blob_util.BlobDef,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    data_format: str = "NDHWC",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Performs the max pooling on the input. Analogous to `tf.nn.max_pool3d <https://www.tensorflow.org/api_docs/python/tf/nn/max_pool3d>`_

    Args:
        input (remote_blob_util.BlobDef):  A 5-D `Blob` of the format specified by data_format.
        ksize (Union[int, Sequence[int]]):  An int or list of ints that has length 1, 3 or 5. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1, 3 or 5. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'`. The padding algorithm
        data_format (str, optional):   "NDHWC" or "NCDHW". Defaults to "NDHWC".
        name (Optional[str], optional): This operator's name(optional).

    Returns:
        remote_blob_util.BlobDef: A `Blob` of format specified by data_format. The max pooled output `Blob`.
    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("MaxPool3D_")
        )
        .Op("max_pool_3d")
        .Input("x", [input])
        .Output("y")
    )
    assert padding in ["VALID", "SAME"]
    op.Attr("padding", padding.lower())
    assert data_format in ["NDHWC", "NCDHW"]
    data_format = "channels_last" if data_format == "NHWC" else "channels_first"
    op.Attr("data_format", data_format)
    pool_size = _GetSequence(ksize, 3, "ksize")
    op.Attr("pool_size", pool_size)
    strides = _GetSequence(strides, 3, "strides")
    op.Attr("strides", strides)
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.avg_pool3d")
def avg_pool3d(
    input: remote_blob_util.BlobDef,
    ksize: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    data_format: str = "NDHWC",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Performs the average pooling on the input. Analogous to `tf.nn.avg_pool3d <https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool3d>`_

    Args:
        input (remote_blob_util.BlobDef): A 5-D `Blob` of shape [batch, height, width, channels].
        ksize (Union[int, Sequence[int]]): An int or list of ints that has length 1, 3 or 5. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1, 3 or 5. The stride of the sliding window for each dimension of the input `Blob`.
        padding (str): '`VALID'` or '`SAME'`. 
        data_format (str, optional):  '`NDHWC'` or '`NCDHW'`. Defaults to "NDHWC".
        name (Optional[str], optional):  This operator's name(optional).Defaults to None.

    Returns:
        remote_blob_util.BlobDef: A `Blob` with the same type as value. The average pooled output `Blob`.
    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("AvgPool3D_")
        )
        .Op("avg_pool_3d")
        .Input("x", [input])
        .Output("y")
    )
    assert padding in ["VALID", "SAME"]
    op.Attr("padding", padding.lower())
    assert data_format in ["NDHWC", "NCDHW"]
    data_format = "channels_last" if data_format == "NHWC" else "channels_first"
    op.Attr("data_format", data_format)
    pool_size = _GetSequence(ksize, 3, "ksize")
    op.Attr("pool_size", pool_size)
    strides = _GetSequence(strides, 3, "strides")
    op.Attr("strides", strides)
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
    r"""Computes softmax activations. Analogous to `tf.nn.softmax <https://www.tensorflow.org/api_docs/python/tf/nn/softmax>`_

    Args:
        logits (remote_blob_util.BlobDef): A non-empty `Blob`. 
        axis (Optional[int], optional): .The dimension softmax would be performed on. Defaults to None.
        name (Optional[str], optional): . This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef:  A `Blob` has the same type and shape as logits.

    Raises:
        InvalidArgumentError: if logits is empty or axis is beyond the last dimension of logits.
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
    r"""Computer sparse cross entropy

    Args:
        labels (remote_blob_util.BlobDef): A `Blob` of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels and result). Each entry in labels must be an index in [0, num_classes).
        prediction (remote_blob_util.BlobDef): A `Blob` with the rank that is equal to the rank of the labels plus one.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef: A `Blob` of the same shape as labels.
    """
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
    r"""Computes softmax cross entropy between logits and labels. Analogous to `tf.nn.softmax_cross_entropy_with_logits <https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits>`_

    Args:
        labels (remote_blob_util.BlobDef): Each vector along the class dimension should hold a valid probability distribution.
        logits (remote_blob_util.BlobDef): Per-label activations, typically a linear output. logits has same shape and dtype as labels.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef: A `Blob` that contains the softmax cross entropy loss. Its type is the same as logits and its shape is the same as labels except that it does not have the last dimension of labels.
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
    r"""Computes sparse softmax cross entropy between logits and labels. Analogous to `tf.nn.sparse_softmax_cross_entropy_with_logits <https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits>`_

    Args:
        labels (remote_blob_util.BlobDef): `Blob` of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels and result). Each entry in labels must be an index in [0, num_classes).
        logits (remote_blob_util.BlobDef): Unscaled log probabilities of shape [d_0, d_1, ..., d_{r-1},num_classes].
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Raises:
        ValueError: If logits are scalars (need to have rank >= 1) or if the rank of the labels is not equal to the rank of the logits minus one.    

    Returns:
        remote_blob_util.BlobDef:  A `Blob` of the same shape as labels and of the same type as logits with the softmax cross entropy loss.
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
    r"""Computes sigmoid cross entropy given logits. Analogous to `tf.nn.sigmoid_cross_entropy_with_logits <https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits>`_

    Args:
        labels (remote_blob_util.BlobDef): A `Blob` of the same type and shape as logits.
        logits (remote_blob_util.BlobDef): A `Blob` of type float.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef:   A `Blob` of the same shape as logits with the componentwise logistic losses.
    
    Raises:
        ValueError: If logits and labels do not have the same shape.
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
        remote_blob_util.BlobDef: A random mask `Blob` of the same shape of '`like'`.
    
    Raises:
        ValueError: If rate is not in [0, 1). Rate=1 is not allowed.
    """
    assert rate is not None and rate >= 0.0 and rate < 1.0
    if noise_shape is not None:
        assert 0, "noise_shape will be supported later."
        assert isinstance(noise_shape, (list, tuple))
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
    r"""For preventing overfitting, randomly set elements to zero. Analogous to `tf.nn.dropout <https://www.tensorflow.org/api_docs/python/tf/nn/dropout>`_

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
        value (Optional[remote_blob_util.BlobDef], optional):   4-d `Blob`. Defaults to None.
        filter (Optional[remote_blob_util.BlobDef], optional): Filter of transposed convolution, usually a variable. Defaults to None.
        output_shape (Optional[remote_blob_util.BlobDef], optional): A 1-D `Blob` representing the output shape of the deconvolution op. Defaults to None.
        strides (Optional[Union[int, Sequence[int]]], optional): `int` or `int list`. Defaults to None.
        padding (str, optional):  `'VALID'` or `'SAME'`. Defaults to "VALID".
        data_format (str, optional): `'NHWC'` or `'NCHW'`. Defaults to "NHWC".
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
    r"""Leaky ReLU activation value

    Args:
        x (remote_blob_util.BlobDef):  A `Blob` representing preactivation values.
        alpha (float, optional): Slope of the activation function at x < 0 with float type. Default value is 0.2.
        name (Optional[str], optional): This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef: The activation `Blob`
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
