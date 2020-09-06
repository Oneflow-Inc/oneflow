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
import numpy as np
import tensorflow as tf

from oneflow.python.ops import nn_ops
from oneflow.python.onnx.load.common import get_data_format
from oneflow.python.onnx.load.common import get_perm_from_formats
from oneflow.python.onnx.load.common import logger


class PoolMixin(object):
    @classmethod
    def pool(cls, node, input_dict, pooling_type, strict=True):
        x = input_dict[node.input_tensor_names[0]]
        orig_x = x

        kernel_shape = node.attrs["kernel_shape"]

        spatial_size = len(kernel_shape)
        x_rank = spatial_size + 2

        kernel_shape = node.attrs["kernel_shape"]
        strides = node.attrs.get("strides", [1] * spatial_size)
        dilations = node.attrs.get("dilations", [1] * spatial_size)
        ceil_mode = bool(node.attrs.get("ceil_mode", 0))
        if ceil_mode != 0:
            raise ValueError("ceil_mode != 0 is not supported")
        pads = node.attrs.get("auto_pad", "NOTSET")
        if pads == "NOTSET":
            pads = node.attrs.get("pads", [0] * spatial_size * 2)
            pads = np.reshape(pads, [2, spatial_size]).T.tolist()
            pads = [[0, 0], [0, 0]] + pads
            # # In case shape is fully defined, check if pads match
            # # SAME padding in Tensorflow
            # if x.shape.is_fully_defined() and pads != [0] * spatial_size * 2:
            #   in_shape = x.get_shape()
            #   same_paddings = calc_pads_same(in_shape[1:x_rank-1], kernel_shape,
            #              strides, dilations, "SAME_UPPER")
            #   if pads == same_paddings:
            #     pads = "SAME_UPPER"

        count_include_pad = bool(node.attrs.get("count_include_pad", 0))
        if count_include_pad != 0:
            raise ValueError("count_include_pad != 0 is not supported")
        if pooling_type == "AVG":
            op = nn_ops.avg_pool2d
        elif pooling_type == "MAX":
            op = nn_ops.max_pool2d
        elif pooling_type == "MAX_WITH_ARGMAX":
            raise ValueError("maxpooling with argmax is not supported")

        if spatial_size != 2:
            raise ValueError("non-2d pooling is not supported")
        if node.attrs.get("storage_order", 0) != 0:
            raise ValueError("storage_order != 0 is not supported")

        return op(
            x, ksize=kernel_shape, strides=strides, padding=pads, data_format="NCHW"
        )
