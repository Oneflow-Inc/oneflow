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
from .pool_mixin import PoolMixin


@onnx_op("MaxPool")
@partial_support(True)
@ps_description(
    "MaxPoolWithArgmax with pad is None or incompatible mode, or "
    + "MaxPoolWithArgmax with 4D or higher input, or"
    + "MaxPoolWithArgmax with column major "
    + "are not supported in Tensorflow."
)
class MaxPool(PoolMixin, BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        pool_type = "MAX" if len(node.output_tensors) == 1 else "MAX_WITH_ARGMAX"
        return cls.pool(
            node, kwargs["tensor_dict"], pool_type, kwargs.get("strict", True)
        )

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_8(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)
