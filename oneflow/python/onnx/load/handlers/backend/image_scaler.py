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

from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op


@onnx_op("ImageScaler")
class ImageScaler(BackendHandler):
    @classmethod
    def version_1(cls, node, **kwargs):
        input_dict = kwargs["tensor_dict"]
        x = input_dict[node.inputs[0]]
        scale = node.attrs.get("scale", 1.0)
        output = tf.multiply(x, scale)
        if "bias" in node.attrs:
            bias = node.attrs["bias"]
            output = tf.nn.bias_add(output, bias, data_format="NCHW")
        return [output]
