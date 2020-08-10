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


@onnx_op("SequenceInsert")
class SequenceInsert(BackendHandler):
    @classmethod
    def chk_pos_in_bounds(cls, input_seq, pos):
        """ 
    Check the position is in-bounds with respect to the sequence.
    Accepted range for 'position' is in [-n, n], where n is the 
    number of tensors in 'input_sequence'. 

    :param input_seq: input sequence
    :param pos: position to insert the tensor

    :return: True if position is in-bounds.
    """
        seq_length = tf.shape(input_seq.to_sparse())[0]
        pos = tf.cast(pos, tf.int32)

        cond1 = tf.greater_equal(pos, tf.negative(seq_length))
        cond2 = tf.less_equal(pos, seq_length)

        # pos >= -n and pos <= n
        return tf.reduce_all(tf.logical_and(cond1, cond2))

    @classmethod
    def version_11(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        input_sequence = tensor_dict[node.inputs[0]]
        input_tensor = tensor_dict[node.inputs[1]]

        position = (
            tensor_dict[node.inputs[2]]
            if len(node.inputs) > 2
            else tf.shape(input_sequence.to_sparse())[0]
        )

        # check whether position is in-bounds and assert if not
        result = cls.chk_pos_in_bounds(input_sequence, position)
        assert_pos = tf.Assert(tf.equal(result, True), [result])

        with tf.control_dependencies([assert_pos]):
            input_tensor = tf.expand_dims(input_tensor, 0)
            if input_sequence.shape[0] == 0:
                output_seq = tf.RaggedTensor.from_tensor(input_tensor)
            else:
                s1 = input_sequence[:position]
                s2 = input_sequence[position:]
                output_seq = tf.concat([s1, input_tensor, s2], axis=0)

            return [output_seq]
