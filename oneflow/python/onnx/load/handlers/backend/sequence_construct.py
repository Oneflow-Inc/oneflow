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


@onnx_op("SequenceConstruct")
class SequenceConstruct(BackendHandler):
    @classmethod
    def version_11(cls, node, **kwargs):
        # create an empty sequence first
        tensor_dict = kwargs["tensor_dict"]
        dtype = tensor_dict[node.inputs[0]].dtype
        input_sequence = tf.ragged.constant([], dtype=dtype)

        # insert tensors at the end of sequence
        for i in range(len(node.inputs)):
            input_tensor = tf.expand_dims(tensor_dict[node.inputs[i]], 0)
            if input_sequence.shape[0] == 0:
                output_seq = tf.RaggedTensor.from_tensor(input_tensor)
            else:
                output_seq = tf.concat([input_sequence, input_tensor], axis=0)
            input_sequence = output_seq

        return [output_seq]
