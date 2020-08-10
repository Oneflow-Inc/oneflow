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

from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.common import data_type
from onnx import mapping


@onnx_op("SequenceEmpty")
class SequenceEmpty(BackendHandler):
    @classmethod
    def version_11(cls, node, **kwargs):
        default_dtype = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype("float32")]
        dtype = data_type.onnx2tf(node.attrs.get("dtype", default_dtype))

        ragged = tf.RaggedTensor.from_row_lengths(values=[], row_lengths=[])
        sparse = tf.cast(ragged.to_sparse(), dtype)
        return [tf.RaggedTensor.from_sparse(sparse)]
