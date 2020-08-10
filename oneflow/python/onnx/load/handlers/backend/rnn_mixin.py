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

# import tensorflow_probability as tfp
from tensorflow.python.ops import array_ops

from oneflow.python.onnx.load.common import exception


class RNNMixin(object):

    ONNX_ACTIVATION_MAPPING = {
        # Added from tf 1.8
        # "affine": tf.contrib.distributions.bijectors.AffineScalar,
        # tf.contrib was removed since tf 2.0,
        # Class Affine had been move to the following module
        # "affine": tfp.bijectors.Affine,
        "elu": tf.nn.elu,
        "hard_sigmoid": tf.keras.backend.hard_sigmoid,
        "leaky_relu": tf.nn.leaky_relu,
        "relu": tf.nn.relu,
        "sigmoid": tf.sigmoid,
        "softsign": tf.nn.softsign,
        "softplus": tf.nn.softplus,
        "tanh": tf.tanh,
        "thresholded_relu": tf.keras.layers.ThresholdedReLU,
    }

    @classmethod
    def rnn(cls, x, cell_class, cell_kwargs, rnn_kwargs, activations, direction):
        cell_kwargs["activation"] = activations[0]

        rnn_cell = [cell_class(**cell_kwargs)]
        cell_fw = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_cell)

        if direction == "bidirectional":
            cell_kwargs["activation"] = activations[1]
            rnn_cell_bw = [cell_class(**cell_kwargs)]
            cell_bw = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_cell_bw)

        if direction == "forward":
            outputs, states = tf.compat.v1.nn.dynamic_rnn(cell_fw, x, **rnn_kwargs)
        elif direction == "bidirectional":
            outputs, states = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, x, **rnn_kwargs
            )
        elif direction == "reverse":

            def _reverse(input_, seq_dim):
                return array_ops.reverse(input_, axis=[seq_dim])

            time_dim = 0
            inputs_reverse = _reverse(x, time_dim)
            outputs, states = tf.compat.v1.nn.dynamic_rnn(
                cell_fw, inputs_reverse, **rnn_kwargs
            )
            outputs = _reverse(outputs, time_dim)

        return outputs, states

    @classmethod
    def rnn_get_activation(cls, name, alpha, beta):
        if name not in cls.ONNX_ACTIVATION_MAPPING:
            exception.OP_UNSUPPORTED_EXCEPT(
                "Activation function {} for {}".format(name, cls.__name__), "Tensorflow"
            )
        activation = cls.ONNX_ACTIVATION_MAPPING[name]
        kwargs = {}
        if name == "affine":
            kwargs["scale"] = alpha
            kwargs["shift"] = beta
            activation = activation(**kwargs)
        elif name == "elu":
            if alpha != 1:
                exception.OP_UNSUPPORTED_EXCEPT(
                    "Activation function {} with alpha={} for {}".format(
                        name, alpha, cls.__name__
                    ),
                    "Tensorflow",
                )
        elif name == "hard_sigmoid":
            if alpha != 0.2 or beta != 0.5:
                exception.OP_UNSUPPORTED_EXCEPT(
                    "Activation function {} with alpha={}, beta={} for {}".format(
                        name, alpha, beta, cls.__name__
                    ),
                    "Tensorflow",
                )
        elif name == "leaky_relu":
            kwargs["alpha"] = alpha or 0.01
            activation = partial(activation, **kwargs)
        elif name == "thresholded_relu":
            kwargs["theta"] = alpha
            activation = activation(**kwargs)
        return activation
