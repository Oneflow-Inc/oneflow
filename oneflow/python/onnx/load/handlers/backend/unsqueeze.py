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


@onnx_op("Unsqueeze")
@tf_func(tf.expand_dims)
class Unsqueeze(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        attrs = copy.deepcopy(node.attrs)
        axes = attrs.pop("axes")
        if len(axes) != 1:
            x = kwargs["tensor_dict"][node.inputs[0]]
            for axis in sorted(axes):
                x = tf.expand_dims(x, axis=axis)
            return [x]
        attrs["axis"] = axes[0]
        return [cls.make_tensor_from_onnx_node(node, attrs=attrs, **kwargs)]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
