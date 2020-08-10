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

import tensorflow as tf

from oneflow.python.onnx.load.common import get_data_format
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import tf_func


@onnx_op("DepthToSpace")
@tf_func(tf.nn.depth_to_space)
class DepthToSpace(BackendHandler):
    @classmethod
    def get_attrs_processor_param(cls):
        return {"rename": {"blocksize": "block_size"}}

    @classmethod
    def version_1(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        x_rank = len(x.get_shape())
        storage_format, compute_format = get_data_format(x_rank)
        attrs = copy.deepcopy(node.attrs)
        attrs["data_format"] = storage_format
        return [
            cls.make_tensor_from_onnx_node(
                node, attrs=attrs, c_first_cuda_only=True, **kwargs
            )
        ]

    @classmethod
    def version_11(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        x_rank = len(x.get_shape())
        storage_format, compute_format = get_data_format(x_rank)
        attrs = copy.deepcopy(node.attrs)
        attrs["data_format"] = storage_format
        mode = attrs.get("mode", "DCR")

        if mode == "CRD":
            # need native computation
            bsize = attrs.get("blocksize")
            batch, channel, height, width = x.shape
            csize = channel // (bsize ** 2)

            reshape_node = tf.reshape(x, [batch, csize, bsize, bsize, height, width])
            transpose_node = tf.transpose(reshape_node, perm=[0, 1, 4, 2, 5, 3])
            return [
                tf.reshape(
                    transpose_node, [batch, csize, height * bsize, width * bsize]
                )
            ]

        else:
            return [
                cls.make_tensor_from_onnx_node(
                    node, attrs=attrs, c_first_cuda_only=True, **kwargs
                )
            ]
