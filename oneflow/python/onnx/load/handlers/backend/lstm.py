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
from functools import partial

import tensorflow as tf

from oneflow.python.onnx.load.common import get_unique_suffix
from oneflow.python.onnx.load.common import exception
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import partial_support
from oneflow.python.onnx.load.handlers.handler import ps_description
from .rnn_mixin import RNNMixin


@onnx_op("LSTM")
@partial_support(True)
@ps_description(
    "LSTM not using sigmoid for `f`, or "
    + "LSTM not using the same activation for `g` and `h` "
    + "are not supported in Tensorflow."
)
class LSTM(RNNMixin, BackendHandler):
    @classmethod
    def args_check(cls, node, **kwargs):
        direction = node.attrs.get("direction", "forward")
        num_directions = 2 if direction == "bidirectional" else 1
        if node.attrs.get("input_forget", 0):
            # TODO(fumihwh): warning
            pass
        if "activations" in node.attrs:
            activations = list(map(lambda x: x.lower(), node.attrs["activations"]))
            if activations[0] != "sigmoid":
                exception.OP_UNSUPPORTED_EXCEPT(
                    "LSTM without sigmoid for `f`", "Tensorflow"
                )
            if activations[1] != activations[2]:
                exception.OP_UNSUPPORTED_EXCEPT(
                    "LSTM without same activation for `g` and `h`", "Tensorflow"
                )
            if num_directions == 2:
                if activations[3] != "sigmoid":
                    exception.OP_UNSUPPORTED_EXCEPT(
                        "LSTM without sigmoid for `f`", "Tensorflow"
                    )
                if activations[4] != activations[5]:
                    exception.OP_UNSUPPORTED_EXCEPT(
                        "LSTM without same activation for `g` and `h`", "Tensorflow"
                    )

    @classmethod
    def _custom_getter(
        cls,
        getter,
        name,
        node=None,
        tensor_dict=None,
        is_bidirectional=None,
        *args,
        **kwargs
    ):
        names = name.split("/")
        if is_bidirectional:
            if "fw" in names:
                index = 0
            elif "bw" in names:
                index = 1
            else:
                raise RuntimeError(
                    "Can not get {} for bidirectional. "
                    "Either fw and bw is not in name scope.".format(names[-1])
                )

        if names[-1] == "kernel":
            # onnx W[iofc], R[iofc]
            if is_bidirectional:
                w = tf.split(tensor_dict[node.inputs[1]], 2)[index]
                r = tf.split(tensor_dict[node.inputs[2]], 2)[index]
            else:
                w = tensor_dict[node.inputs[1]]
                r = tensor_dict[node.inputs[2]]
            w_i, w_o, w_f, w_c = tf.split(tf.squeeze(w), 4)
            r_i, r_o, r_f, r_c = tf.split(tf.squeeze(r), 4)
            new_w = tf.transpose(tf.concat([w_i, w_c, w_f, w_o], 0))
            new_r = tf.transpose(tf.concat([r_i, r_c, r_f, r_o], 0))
            kernel = tf.concat([new_w, new_r], 0)
            return kernel
        if names[-1] == "bias":
            if len(node.inputs) >= 4:
                # onnx Wb[iofc], Rb[iofc]
                if is_bidirectional:
                    b = tf.split(tensor_dict[node.inputs[3]], 2)[index]
                else:
                    b = tensor_dict[node.inputs[3]]
                w_b, r_b = tf.split(tf.squeeze(b), 2)
                w_b_i, w_b_o, w_b_f, w_b_c = tf.split(w_b, 4)
                r_b_i, r_b_o, r_b_f, r_b_c = tf.split(r_b, 4)
                w_b = tf.transpose(tf.concat([w_b_i, w_b_c, w_b_f, w_b_o], 0))
                r_b = tf.transpose(tf.concat([r_b_i, r_b_c, r_b_f, r_b_o], 0))
                return tf.add(w_b, r_b)
            return getter(name, *args, **kwargs)
        # Only use_peepholes is True,
        # will try to get w_f_diag, w_i_diag, w_o_diag
        # onnx P[iof]
        if names[-1] in ["w_f_diag", "w_i_diag", "w_o_diag"]:
            if is_bidirectional:
                p = tf.split(tensor_dict[node.inputs[7]], 2)[index]
            else:
                p = tensor_dict[node.inputs[7]]
            if names[-1] == "w_f_diag":
                return tf.split(p, 3, axis=1)[2]
            if names[-1] == "w_i_diag":
                return tf.split(p, 3, axis=1)[0]
            if names[-1] == "w_o_diag":
                return tf.split(p, 3, axis=1)[1]
        return getter(name, *args, **kwargs)

    @classmethod
    def _common(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]
        input_shape = x.get_shape().as_list()
        input_size = len(node.inputs)
        hidden_size = node.attrs["hidden_size"]
        direction = node.attrs.get("direction", "forward")
        num_directions = 2 if direction == "bidirectional" else 1

        # removed from version 7, default is 0
        output_sequence = node.attrs.get("output_sequence", 0)

        # TODO(fumihwh): check if prev node is one of RNN
        # process input if it comes from other previous cell
        # which has shape [seq_length, num_directions, batch_size, hidden_size]
        if len(input_shape) == 4 and input_shape[1] == 1:
            x = tf.squeeze(x)

        sequence_length = None
        if input_size >= 5 and node.inputs[4] in tensor_dict:
            sequence_length = tensor_dict[node.inputs[4]]

        cell_kwargs = {}

        if "clip" in node.attrs:
            cell_kwargs["cell_clip"] = node.attrs["clip"]

        tf_activations = [tf.nn.tanh] * num_directions
        if "activations" in node.attrs:
            activations = list(map(lambda x: x.lower(), node.attrs["activations"]))
            activation_alpha = node.attrs.get("activation_alpha", [None] * 6)
            activation_beta = node.attrs.get("activation_beta", [None] * 6)

            # tf only supports cutomizing hidden states activation function,
            # which correspond to activation functions specified at position 1
            # and 4 in onnx's activations attribute.
            activation_idxs = [1, 4] if num_directions == 2 else [1]
            tf_activations = [
                cls.rnn_get_activation(
                    activations[i], activation_alpha[i], activation_beta[i]
                )
                for i in activation_idxs
            ]

        # TODO(fumihwh): check if reverse and bidirectional works
        with tf.compat.v1.variable_scope(
            "LSTM_" + get_unique_suffix(),
            custom_getter=partial(
                cls._custom_getter,
                node=node,
                tensor_dict=tensor_dict,
                is_bidirectional=num_directions == 2,
            ),
        ):

            cell_kwargs["use_peepholes"] = (
                input_size == 8 and node.inputs[7] in tensor_dict
            )
            cell_kwargs["forget_bias"] = 0.0
            cell_kwargs["num_units"] = hidden_size
            initial_state = None
            initial_state_bw = None
            if input_size >= 6:
                initial_h = tensor_dict.get(node.inputs[5], None)
                initial_c = (
                    tensor_dict.get(node.inputs[6], None)
                    if input_size >= 7
                    else tf.zeros_like(initial_h)
                )
                if initial_h is not None and initial_c is not None:
                    initial_state = (
                        tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
                            initial_c[0], initial_h[0]
                        ),
                    )
                    if num_directions == 2:
                        initial_state_bw = (
                            tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
                                initial_c[1], initial_h[1]
                            ),
                        )

            rnn_kwargs = {}
            if num_directions == 1:
                rnn_kwargs["initial_state"] = initial_state
            elif num_directions == 2:
                rnn_kwargs["initial_state_fw"] = initial_state
                rnn_kwargs["initial_state_bw"] = initial_state_bw
            rnn_kwargs["sequence_length"] = sequence_length
            rnn_kwargs["time_major"] = True
            rnn_kwargs["dtype"] = tf.float32

            outputs, states = cls.rnn(
                x,
                tf.compat.v1.nn.rnn_cell.LSTMCell,
                cell_kwargs,
                rnn_kwargs,
                tf_activations,
                direction,
            )

        if num_directions == 1:
            state = states[0]
            c = tf.expand_dims(state[0], 0)
            h = tf.expand_dims(state[1], 0)
            output = tf.expand_dims(outputs, 1)
        else:
            state_fw = states[0][0]
            state_bw = states[1][0]
            output_fw = outputs[0]
            output_bw = outputs[1]
            c_fw = tf.expand_dims(state_fw[0], 0)
            c_bw = tf.expand_dims(state_bw[0], 0)
            c = tf.concat((c_fw, c_bw), axis=0)
            h_fw = tf.expand_dims(state_fw[1], 0)
            h_bw = tf.expand_dims(state_bw[1], 0)
            h = tf.concat((h_fw, h_bw), axis=0)
            output_fw = tf.expand_dims(output_fw, 1)
            output_bw = tf.expand_dims(output_bw, 1)
            output = tf.concat((output_fw, output_bw), axis=1)

        return [output, h, c] if output_sequence == 0 else [h, c]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)
