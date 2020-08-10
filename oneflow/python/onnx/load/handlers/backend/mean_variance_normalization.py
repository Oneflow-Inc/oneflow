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


@onnx_op("MeanVarianceNormalization")
class MeanVarianceNormalization(BackendHandler):
    @classmethod
    def version_1(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        inputs = tensor_dict[node.inputs[0]]
        inputs_rank = inputs.shape.ndims

        across_channels = node.attrs.get("across_channels", 0)
        normalize_variance = node.attrs.get("normalize_variance", 1)

        moments_axes = [0] if not across_channels else [0, 1]
        moments_axes += list(range(inputs_rank))[2:]

        mean, variance = tf.nn.moments(inputs, moments_axes, keepdims=True)

        if not normalize_variance:
            return [inputs - mean]
        return [(inputs - mean) / tf.sqrt(variance)]

    @classmethod
    def version_9(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        inputs = tensor_dict[node.inputs[0]]
        inputs_rank = inputs.shape.ndims
        # To satisfy default axes=[0,2,3], also assume the
        # following default when rank is not 4
        # rank1 -> axes=[0]
        # rank2 -> axes=[0]
        # rank3 -> axes=[0,2]
        # rank4 -> axes=[0,2,3]
        # rankN -> axes=[0,2,3,..,N-1]
        # TODO(tedhtchang): Since input tensor is no longer limited
        # to shape [N,C,H,W], consider using "[0]" or "[]" as default axes.
        # See issue https://github.com/onnx/onnx/issues/2047
        default_axes = [0] if inputs_rank < 3 else [0, 2]
        default_axes += list(range(inputs_rank))[3:]
        moments_axes = node.attrs.get("axes", default_axes)
        mean, variance = tf.nn.moments(inputs, moments_axes, keepdims=True)
        return [(inputs - mean) / tf.sqrt(variance)]
