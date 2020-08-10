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
import numpy as np

from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import tf_func


@onnx_op("LRN")
@tf_func(tf.nn.lrn)
class LRN(BackendHandler):
    @classmethod
    def version_1(cls, node, **kwargs):
        attrs = copy.deepcopy(node.attrs)
        alpha = attrs.get("alpha", 1e-4)
        attrs.setdefault("beta", 0.75)
        size = attrs["size"]
        attrs["alpha"] = alpha / size
        attrs["depth_radius"] = np.floor([(size - 1) / 2.0])[0]
        # TODO: LRN in tf accepts radius
        # but in ONNX/Caffe accepts diameter.
        # This could be a problem.
        return [
            cls.make_tensor_from_onnx_node(
                node, attrs=attrs, c_last_only=True, **kwargs
            )
        ]
