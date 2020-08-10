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

from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import tf_func
from oneflow.python.onnx.load.common.tf_helper import tf_shape


@onnx_op("Split")
@tf_func(tf.split)
class Split(BackendHandler):
    @classmethod
    def args_check(cls, node, **kwargs):
        axis = node.attrs.get("axis", 0)
        x_rank = len(kwargs["tensor_dict"][node.inputs[0]].get_shape().as_list())
        if axis > x_rank - 1 or axis < -x_rank:
            raise ValueError("Axis is out of bound")

    @classmethod
    def get_attrs_processor_param(cls):
        return {"default": {"axis": 0}}

    @classmethod
    def _common(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        input = tensor_dict[node.inputs[0]]
        x_shape = tf_shape(input)
        attrs = copy.deepcopy(node.attrs)
        axis = attrs.get("axis", 0)
        axis = axis if axis >= 0 else len(x_shape) + axis
        if "split" in node.attrs:
            split = attrs["split"]
        elif len(node.inputs) == 2:  # since version 1
            split = tensor_dict[node.inputs[1]]
        else:
            per_part = x_shape[axis] / len(node.outputs)
            if input.shape.is_fully_defined():
                if int(per_part) != per_part:
                    raise ValueError("Split can not be evenly divided.")
                split = [int(per_part)] * len(node.outputs)
            else:
                split = [tf.cast(per_part, tf.int32)] * len(node.outputs)
        attrs["num_or_size_splits"] = split
        return list(
            cls.make_tensor_from_onnx_node(node, inputs=[input], attrs=attrs, **kwargs)
        )

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_2(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
