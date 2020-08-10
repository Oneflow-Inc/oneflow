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
import numpy as np


def flow_shape(blob):
    return blob.shape


def tf_shape(tensor):
    """
        Helper function returning the shape of a Tensor.
        The function will check for fully defined shape and will return
        numpy array or if the shape is not fully defined will use tf.shape()
        to return the shape as a Tensor.
    """
    if tensor.shape.is_fully_defined():
        return np.array(tensor.shape.as_list(), dtype=np.int64)
    else:
        return tf.shape(tensor, out_type=tf.int64)


def tf_product(a, b):
    """
        Calculates the cartesian product of two column vectors a and b

        Example:

        a = [[1]
             [2]
             [3]]

        b = [[0]
             [1]]

        result = [[1 0]
                  [1 1]
                  [2 0]
                  [2 1]
                  [3 0]
                  [3 1]]
    """
    tile_a = tf.tile(a, [1, tf.shape(b)[0]])
    tile_a = tf.expand_dims(tile_a, 2)
    tile_a = tf.reshape(tile_a, [-1, 1])

    b = tf.tile(b, [tf.shape(a)[0], 1])
    b = tf.concat([tile_a, b], axis=1)

    return b
