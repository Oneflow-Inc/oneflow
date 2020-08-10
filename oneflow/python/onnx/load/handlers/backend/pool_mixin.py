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
import tensorflow as tf

from oneflow.python.onnx.load.common import exception
from oneflow.python.onnx.load.common import get_data_format
from oneflow.python.onnx.load.common import get_perm_from_formats
from oneflow.python.onnx.load.common import logger
from .dilated_pooling import DilatedPooling
from oneflow.python.onnx.load.common.pooling_helper import py_pool
from oneflow.python.onnx.load.common.pooling_helper import calc_pads_same
from oneflow.python.onnx.load.common.pooling_helper import calc_output_shape


class PoolMixin(object):
    @classmethod
    def pool(cls, node, input_dict, pooling_type, strict=True):
        x = input_dict[node.inputs[0]]
        orig_x = x

        kernel_shape = node.attrs["kernel_shape"]

        spatial_size = len(kernel_shape)
        x_rank = spatial_size + 2

        kernel_shape = node.attrs["kernel_shape"]
        strides = node.attrs.get("strides", [1] * spatial_size)
        dilations = node.attrs.get("dilations", [1] * spatial_size)
        ceil_mode = bool(node.attrs.get("ceil_mode", 0))
        pads = node.attrs.get("auto_pad", "NOTSET")
        if pads == "NOTSET":
            pads = node.attrs.get("pads", [0] * spatial_size * 2)
            # # In case shape is fully defined, check if pads match
            # # SAME padding in Tensorflow
            # if x.shape.is_fully_defined() and pads != [0] * spatial_size * 2:
            #   in_shape = x.get_shape()
            #   same_paddings = calc_pads_same(in_shape[1:x_rank-1], kernel_shape,
            #              strides, dilations, "SAME_UPPER")
            #   if pads == same_paddings:
            #     pads = "SAME_UPPER"

        count_include_pad = bool(node.attrs.get("count_include_pad", 0))
        if pooling_type == "AVG":
            pooling_name = "AveragePool"
        elif pooling_type == "MAX":
            pooling_name = "MaxPool"
        elif pooling_type == "MAX_WITH_ARGMAX":
            pooling_name = "MaxPoolWithArgmax"

        if spatial_size > 3:
            exception.OP_UNSUPPORTED_EXCEPT(
                pooling_name + " with {}D input".format(x_rank), "Tensorflow"
            )
        if pooling_type == "MAX_WITH_ARGMAX" and x_rank != 4:
            exception.OP_UNSUPPORTED_EXCEPT(
                pooling_name + " with {}D input".format(x_rank), "Tensorflow"
            )
        if node.attrs.get("storage_order", 0) != 0:
            exception.OP_UNSUPPORTED_EXCEPT(
                pooling_name + " with column major", "Tensorflow"
            )

        dp = DilatedPooling(
            input=x,
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
            padding=pads,
            ceil_mode=ceil_mode,
            pooling_type=pooling_type,
            count_include_pad=count_include_pad,
        )
        if not dp.is_supported():
            if strict:
                logger.warning(
                    "Using the pooling op in compatibility mode. "
                    "This means your graph cannot be serialized.",
                    UserWarning,
                )

                result = tf.numpy_function(
                    py_pool,
                    [
                        orig_x,
                        kernel_shape,
                        strides,
                        dilations,
                        pads,
                        ceil_mode,
                        "AVG",
                        False,
                    ],
                    orig_x.dtype,
                )

                if orig_x.shape.is_fully_defined():
                    shape = orig_x.get_shape()
                    output_shape = shape[0:2] + calc_output_shape(
                        shape[2:x_rank],
                        kernel_shape,
                        strides,
                        dilations,
                        pads,
                        ceil_mode,
                    )
                else:
                    output_shape = [None] * x_rank
                result.set_shape(output_shape)
                return [result]
            else:
                exception.OP_UNSUPPORTED_EXCEPT(
                    "strict == 0 and average pool" " arguments not compatible",
                    "Tensorflow",
                )

        def dilated_pool():
            return (dp.dilated_pool(), None)

        # select correct op depending on the pooling type
        pooling_op = (
            dilated_pool
            if pooling_type in ["MAX", "AVG"]
            else dp.dilated_maxpool_with_argmax
        )

        pooled, argmax = pooling_op()

        result = [pooled] if argmax is None else [pooled, argmax]

        return result
