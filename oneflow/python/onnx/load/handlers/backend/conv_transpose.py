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
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import partial_support
from oneflow.python.onnx.load.handlers.handler import ps_description
from .conv_mixin import ConvMixin


@onnx_op("ConvTranspose")
@partial_support(True)
@ps_description(
    "ConvTranspose with dilations != 1, or "
    + "transposed convolution for 4D or higher "
    + "are not supported in Tensorflow."
)
class ConvTranspose(ConvMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, **kwargs):
        return cls.conv(node, kwargs["tensor_dict"], transpose=True)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls.conv(node, kwargs["tensor_dict"], transpose=True)
