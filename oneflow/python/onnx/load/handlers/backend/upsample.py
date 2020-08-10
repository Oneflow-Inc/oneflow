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
import copy

import numpy as np
import tensorflow as tf

from oneflow.python.onnx.load.common import exception
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import partial_support
from oneflow.python.onnx.load.handlers.handler import ps_description
from oneflow.python.onnx.load.handlers.handler import tf_func


@onnx_op("Upsample")
@tf_func(tf.image.resize)
@partial_support(True)
@ps_description("Upsample required 4D input in Tensorflow.")
class Upsample(BackendHandler):
    @classmethod
    def args_check(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        x_shape = x.get_shape().as_list()
        if len(x_shape) != 4:
            exception.OP_UNSUPPORTED_EXCEPT("Upsample without 4D input", "Tensorflow")

        if node.attrs.get("mode", "nearest").lower() not in [
            "nearest",
            "bilinear",
            "linear",
        ]:
            exception.OP_UNSUPPORTED_EXCEPT(
                "Upsample without nearest or bilinear", "Tensorflow"
            )

    @classmethod
    def version_7(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        x_shape = x.get_shape().as_list()
        attrs = copy.deepcopy(node.attrs)
        scales = attrs["scales"]
        new_height = np.floor(x_shape[2] * scales[2])
        new_weight = np.floor(x_shape[3] * scales[3])

        mode = attrs.get("mode", "nearest")
        if mode.lower() == "bilinear" or mode.lower() == "linear":
            mode = tf.image.ResizeMethod.BILINEAR
        else:
            mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR

        attrs["size"] = np.array((new_height, new_weight), dtype=np.int32)
        attrs["method"] = mode

        return [
            cls.make_tensor_from_onnx_node(
                node, attrs=attrs, c_last_only=True, **kwargs
            )
        ]

    @classmethod
    def version_9(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        x_shape = x.get_shape().as_list()
        attrs = copy.deepcopy(node.attrs)
        scales = kwargs["tensor_dict"][node.inputs[1]]

        assert_n_c_scale_is_one = tf.Assert(
            tf.logical_and(tf.equal(scales[0], 1), tf.equal(scales[1], 1)), [scales]
        )

        with tf.control_dependencies([assert_n_c_scale_is_one]):
            h_w_scale = scales[2:]
            h_w_shape = x_shape[2:]
            new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

            mode = attrs.get("mode", "nearest")
            if mode.lower() == "bilinear" or mode.lower() == "linear":
                mode = tf.image.ResizeMethod.BILINEAR
            else:
                mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR

            attrs["size"] = new_h_w_shape
            attrs["method"] = mode

            # Remove scale.
            upsample_node = copy.deepcopy(node)
            del upsample_node.inputs[1]
            return [
                cls.make_tensor_from_onnx_node(
                    upsample_node, attrs=attrs, c_last_only=True, **kwargs
                )
            ]
