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
from oneflow.python.ops.nn_ops import calc_pool_padding, _GetSequence

IntPair = Tuple[int, int]

def calc_unfold_padding(padding, dhw_offset, ndims):
    if isinstance(padding, int):
        padding = _GetSequence(padding, ndims, "padding")
    return calc_pool_padding(padding, dhw_offset, ndims)


@oneflow_export("nn.unfold1d")
def unfold1d(
    input: remote_blob_util.BlobDef,
    kernel_size: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    dilation_rate: Union[int, Sequence[int]],
    padding: Union[str, int, Sequence[Sequence[int]]],
    data_format: str = "NCW",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Performs the unfold on the input `Blob`.

    Args:
        input (remote_blob_util.BlobDef): A 3-D `Blob` of the format specified by data_format.
        kernel_size (Union[int, Sequence[int]]): An int or list of ints that has length 1. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 1. The stride of the sliding window for each dimension of the input `Blob`.
        dilation_rate (Union[int, Sequence[int]]): An int or list of ints that has length 1. The dilation_rate of the sliding window for each dimension of the input `Blob`.
        padding (Union[str, int, Sequence[Sequence[int]]]): '`VALID'` or '`SAME'` or '`SAME_LOWER'`, or '`int'`, or '`SAME_UPPER or Sequence[Sequence[int]]'`.
        data_format (str, optional): '`NCW'`.
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Raises:
        NotImplementedError: not implement unfold1d currently

    Returns:
        remote_blob_util.BlobDef: A `Blob` of format specified by data_format. The unfold output `Blob`.
    """
    raise NotImplementedError


@oneflow_export("nn.unfold2d")
def unfold2d(
    input: remote_blob_util.BlobDef,
    kernel_size: Union[int, IntPair],
    strides: Union[int, IntPair],
    dilation_rate: Union[int, IntPair],
    padding: Union[str, int, Sequence[Sequence[int]]],
    data_format: str = "NCHW",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Performs the unfold on the input. 

    Args:
        input (remote_blob_util.BlobDef): A 4-D `Blob` of shape [batch, height, width, channels].
        kernel_size (Union[int, IntPair]):  An int or list of ints that has length 2. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, IntPair]): An int or list of ints that has length 2. The stride of the sliding window for each dimension of the input `Blob`.
        dilation_rate (Union[int, IntPair]): An int or list of ints that has length 2. The dilation_rate of the sliding window for each dimension of the input `Blob`.
        padding (Union[str, int, Sequence[Sequence[int]]]): '`VALID'` or '`SAME'` or '`SAME_LOWER'`, or '`int'`, or '`SAME_UPPER or Sequence[Sequence[int]]'`.
        data_format (str, optional): '`NCHW'`.
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Returns:
        remote_blob_util.BlobDef:  A `Blob` with the same type as '`value'`. The unfold output `Blob`.
    
    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def unfold2d_Job(x: tp.Numpy.Placeholder((1, 3, 8, 8))
        ) -> tp.Numpy:
            unfold_out = flow.nn.unfold2d(
                input=x,
                kernel_size=3,
                strides=2,
                dilation_rate=1,
                padding='SAME',
                data_format='NCHW'
            )

            return unfold_out


        x = np.random.randn(1, 3, 8, 8).astype(np.float32)
        out = unfold2d_Job(x)

        # out.shape (1, 27, 16)

    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Unfold2D_")
        )
        .Op("unfold_2d")
        .Input("x", [input])
        .Output("y")
    )
    assert data_format in ["NHWC", "NCHW"]
    channel_pos = "channels_last" if data_format == "NHWC" else "channels_first"
    op.Attr("data_format", channel_pos)
    kernel_size = _GetSequence(kernel_size, 2, "kernel_size")
    op.Attr("kernel_size", kernel_size)
    strides = _GetSequence(strides, 2, "strides")
    op.Attr("strides", strides)
    dilation_rate = _GetSequence(dilation_rate, 2, "dilation_rate")
    op.Attr("dilation_rate", dilation_rate)
    padding_type, pads_list = calc_unfold_padding(padding, 0, 2)
    assert len(pads_list) == len(input.shape) - 2
    padding_before = [pad[0] for pad in pads_list]
    padding_after = [pad[1] for pad in pads_list]
    op.Attr("padding", padding_type)
    op.Attr("padding_before", padding_before)
    op.Attr("padding_after", padding_after)
    op.Attr("ceil_mode", ceil_mode)
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.unfold3d")
def unfold3d(
    input: remote_blob_util.BlobDef,
    kernel_size: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    dilation_rate: Union[int, Sequence[int]],
    padding: Union[str, int, Sequence[Sequence[int]]],
    data_format: str = "NCDHW",
    ceil_mode: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Performs the unfold on the input. 

    Args:
        input (remote_blob_util.BlobDef): A 5-D `Blob` of shape [batch, height, width, channels].
        kernel_size (Union[int, Sequence[int]]): An int or list of ints that has length 3. The size of the window for each dimension of the input `Blob`.
        strides (Union[int, Sequence[int]]): An int or list of ints that has length 3. The stride of the sliding window for each dimension of the input `Blob`.
        dilation_rate (Union[int, Sequence[int]]): An int or list of ints that has length 3. The dilation_rate of the sliding window for each dimension of the input `Blob`.
        padding (Union[str, int, Sequence[Sequence[int]]]): '`VALID'` or '`SAME'` or '`SAME_LOWER'`, or '`int'`, or '`SAME_UPPER or Sequence[Sequence[int]]'`.
        data_format (str, optional): '`NCDHW'`.
        name (Optional[str], optional):  This operator's name(optional).Defaults to None.

    Raises:
        NotImplementedError: not implement unfold3d currently

    Returns:
        remote_blob_util.BlobDef: A `Blob` with the same type as value. The unfold output `Blob`.
    """
    raise NotImplementedError
