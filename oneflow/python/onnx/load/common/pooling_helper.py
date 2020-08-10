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
from __future__ import division

from collections import namedtuple
import numpy as np
import tensorflow as tf

import itertools


pad_ops = namedtuple("pad_ops", ["max_op", "ceil_op", "floor_op", "cast_int_op"])

pad_numpy_ops = pad_ops(np.maximum, np.ceil, np.floor, lambda arr: arr.astype(np.int64))
pad_tf_ops = pad_ops(
    tf.maximum, tf.math.ceil, tf.math.floor, lambda tensor: tf.cast(tensor, tf.int64)
)


def calc_pads_same(
    in_spatial_shape,
    kernel_shape,
    strides,
    dilations,
    padding,
    padding_ops=pad_numpy_ops,
    pads_order=1,
):
    """
        Calculates the SAME paddings that need to be added to the input

        Args:
            in_spatial_shape:   input spatial shape
            kernel_shape:       the size of the kernel along each axis
            strides:            stride along each spatial axis
            dilations:          dilations value along each spatial axis
            padding:            padding to calculate: SAME_UPPER or
                                SAME_LOWER
            padding_ops:        namedtuple with ops to be used during
                                calculations. there are two sets of ops
                                defined pad_numpy_ops and pad_tf_ops with
                                numpy and tensorflow ops
            pads_order:         order of returned pads. possible options are:
                                    1 - b1, b2, ..., bn, e1, e2, ..., en
                                    2 - b1, e1, b2, e2, ..., bn, en
                                where n = len(kernel_shape) * 2,
                                b1, b2, ..., bn define pads at the begging of
                                                axis
                                e1, e2, ..., en define pads at the end of
                                                axis
        Return:
            pads:               array with calculated pads. the order of the
                                values is determined by `pads_order`

    """
    spatial_size = len(kernel_shape)
    pads = [0] * (spatial_size * 2)
    for i in range(spatial_size):
        in_size = in_spatial_shape[i]
        filter_size = (kernel_shape[i] - 1) * dilations[i] + 1

        out_size = padding_ops.ceil_op(in_size / strides[i])
        out_size = padding_ops.cast_int_op(out_size)
        pad_along_axis = padding_ops.max_op(
            (out_size - 1) * strides[i] + filter_size - in_size, 0
        )
        if padding.lower() == "same_lower":
            pad_op = padding_ops.ceil_op
        else:
            pad_op = padding_ops.floor_op
        pad_begin = pad_op(pad_along_axis / 2)

        pad_begin = padding_ops.cast_int_op(pad_begin)
        pad_along_axis = padding_ops.cast_int_op(pad_along_axis)

        pad_end = pad_along_axis - pad_begin

        pads[i * pads_order] = pad_begin
        pads[i * pads_order + (spatial_size if pads_order == 1 else 1)] = pad_end

    return pads


def calc_output_shape(
    input_spatial_shape, kernel_shape, strides, dilations, padding, ceil_mode=False
):
    """
        Calculate output shape

        Args:
            input_spatial_shape: input spatial shape
            kernel_shape:        the size of the kernel along each axis
            strides:             stride along each spatial axis
            dilations:           dilations value along each spatial axis
            padding:             can be explicit paddings, "SAME_UPPER" or
                                 "SAME_LOWER"
        Return:
            output_shape:        calculated output shape
    """
    spatial_size = len(input_spatial_shape)

    if type(padding) is not list and type(padding) is not np.ndarray:
        if padding.lower().startswith("same"):
            padding = calc_pads_same(
                input_spatial_shape, kernel_shape, strides, dilations, padding
            )
        else:
            padding = [0] * spatial_size * 2

    output_shape = []
    for dim in range(spatial_size):
        output_shape.append(
            _pooling_output_shape(
                input_spatial_shape[dim],
                kernel_shape[dim],
                strides[dim],
                dilations[dim],
                padding[dim] + padding[dim + spatial_size],
                ceil_mode,
            )
        )

    return output_shape


def _pooling_output_shape(input_size, ksize, stride, dilation, pad, ceil_mode):
    output_size = (
        input_size
        + pad
        - ((ksize - 1) * dilation + 1)
        + ((stride - 1) if ceil_mode else 0)
    ) // stride + 1
    if pad:
        if (output_size - 1) * stride >= input_size + pad:
            output_size -= 1
    return output_size


def py_pool(
    input,
    kernel_shape,
    strides=None,
    dilations=None,
    padding=None,
    ceil_mode=False,
    pooling_type="MAX",
    include_indices=True,
):
    """
        Implementation of Max and Average pool operations in Python
        Args:
            input:        input N-D data array in NC* format
            kernel_shape: the size of the kernel along each axis
            strides:      stride along each spatial axis
            dilations:    dilations value along each spatial axis of filter
            padding:      padding for the beginning and ending along each
                          spatial axis. `padding` format should be as follow
                          [x1_begin, x2_begin...x1_end, x2_end,...]
            ceil_mode:    whether to use ceil or floor (default) to compute
                          the output shape.
            pooling_type: specify pooling type. Values can be "MAX" or "AVG".
            include_indices: should indices be included in the output
      Return:
            pooled:       output data from max pooling across the input
            ind:          indices of the selected max values from the input
    """

    if type(pooling_type) is not str:
        pooling_type = pooling_type.decode("UTF-8")

    input_shape = np.shape(input)
    inp_sp_shape = input_shape[2:]
    input_dtype = input.dtype
    if np.issubdtype(input_dtype, np.integer):
        input_dtype_min = np.iinfo(input_dtype).min
    else:
        input_dtype_min = np.finfo(input_dtype).min

    def _loop_over_output(batch, channel):
        dims = [range(output_sp_shape[d]) for d in range(spatial_size)]
        for counters in itertools.product(*dims):
            input_ranges = []
            for dim in range(spatial_size):
                dim_start = counters[dim] * strides[dim] - pads[dim * 2]
                dim_end = min(
                    dim_start + (kernel_shape[dim] - 1) * dilations[dim] + 1,
                    inp_sp_shape[dim],
                )
                while dim_start < 0:
                    dim_start += dilations[dim]

                cur_range = [i for i in range(dim_start, dim_end, dilations[dim])]
                input_ranges.append(cur_range)
            if pooling_type == "AVG":
                val_sum = 0
                val_count = 0
            else:
                maxval = input_dtype_min
                maxind = -1
            for input_ind in itertools.product(*input_ranges):
                ind = (batch, channel) + input_ind
                val = input[ind]
                if pooling_type == "AVG":
                    val_sum += val
                    val_count += 1
                else:
                    if val > maxval:
                        maxval = val
                        ind = 0
                        for i in range(spatial_size):
                            coef = 1
                            for j in range(i + 1, spatial_size):
                                coef *= inp_sp_shape[j]
                            ind += input_ind[i] * coef
                        maxind = ind
            ind = (batch, channel) + counters
            if pooling_type == "AVG":
                out_pool[ind] = val_sum / val_count
            else:
                out_pool[ind] = maxval
                out_ind[ind] = maxind

    spatial_size = len(kernel_shape)

    batch_size = input_shape[0]
    channels_num = input_shape[1]

    if strides is None:
        strides = kernel_shape

    if dilations is None:
        dilations = [1] * spatial_size

    if padding is None:
        padding = [0] * spatial_size * 2

    if type(padding) is bytes:
        padding = padding.decode()

    if type(padding) is not list and type(padding) is not np.ndarray:
        if padding.lower().startswith("same"):
            padding = calc_pads_same(
                inp_sp_shape, kernel_shape, strides, dilations, padding
            )
        else:
            padding = [0] * spatial_size * 2

    pads = []
    pad_along_axis = []
    output_sp_shape = []

    for dim in range(spatial_size):
        pads.append(padding[dim])
        pads.append(padding[dim + spatial_size])
        pad_along_axis.append(padding[dim] + padding[dim + spatial_size])

        input_size = input_shape[dim + 2]
        output_size = _pooling_output_shape(
            input_size,
            kernel_shape[dim],
            strides[dim],
            dilations[dim],
            pad_along_axis[dim],
            ceil_mode,
        )
        output_sp_shape.append(output_size)

    out_pool = np.zeros([input_shape[0], input_shape[1]] + output_sp_shape, input_dtype)
    out_ind = np.zeros([input_shape[0], input_shape[1]] + output_sp_shape, np.int64)

    for batch in range(batch_size):
        for channel in range(channels_num):
            _loop_over_output(batch, channel)

    if not include_indices:
        return out_pool
    else:
        return out_pool, out_ind
