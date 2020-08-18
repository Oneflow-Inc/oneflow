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

from oneflow.python.ops import layers
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import tf_func
import string
import random


@onnx_op("BatchNormalization")
@tf_func(layers.batch_normalization)
class BatchNormalization(BackendHandler):
    @classmethod
    def get_attrs_processor_param(cls):
        return {
            "default": {"epsilon": 1e-5},
        }

    @classmethod
    def _common(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]

        def randomString(stringLength=8):
            letters = string.ascii_lowercase
            return "".join(random.choice(letters) for i in range(stringLength))

        name = "bn_" + randomString()

        cls.copy_variable_file(node.input_tensor_names[1], name + "-gamma")
        cls.copy_variable_file(node.input_tensor_names[2], name + "-beta")
        cls.copy_variable_file(node.input_tensor_names[3], name + "-moving_mean")
        cls.copy_variable_file(node.input_tensor_names[4], name + "-moving_variance")
        node.input_tensor_names = node.input_tensor_names[:1]

        return [
            cls.make_tensor_from_onnx_node(node, name=name, **kwargs, attrs={"axis": 1})
        ]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)
