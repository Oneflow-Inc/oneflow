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

from oneflow.python.onnx.load.common.tf_helper import tf_shape


class GatherAndScatterMixin(object):
    @classmethod
    def chk_idx_out_of_bounds(cls, data, indices):
        """ Check indices out of bounds for ScatterND and GatherND
    In Tensorflow GPU version, if an out of bound index is found,
    a 0 is stored in the corresponding output value for GatherND;
    and the index is ignored for ScatterND/TensorScatterNDUpdate.
    But ONNX spec state that it is an error if any index values
    are out of bounds. Therefore the converter need to run this
    function to verify all the indices are in bounds before send
    it to Tensoflow. If out of bound is detected then the caller
    of this function need to throw InvalidArgumentError exception.
    """
        data_shape = tf_shape(data)
        indices_shape = tf_shape(indices)

        def _chk_idx_out_of_bounds(i, result):
            indices_i = tf.transpose(indices)[i]
            limit_i = tf.cast(data_shape, indices.dtype)[i]
            cond1 = tf.greater_equal(indices_i, tf.negative(limit_i))
            cond2 = tf.less(indices_i, limit_i)
            result = tf.reduce_all(tf.logical_and(cond1, cond2))
            return i + 1, result

        _, result = tf.while_loop(
            lambda i, result: tf.logical_and(tf.less(i, indices_shape[-1]), result),
            _chk_idx_out_of_bounds,
            [tf.zeros([], tf.int64), True],
        )
        return result

    @classmethod
    def chk_idx_out_of_bounds_along_axis(cls, data, axis, indices):
        """ Check indices out of bounds for ScatterElement
    In Tensorflow GPU version, if an out of bound index is found,
    the index is ignored for ScatterND/TensorScatterNDUpdate.
    But ONNX spec state that it is an error if any index values
    are out of bounds. Therefore the converter need to run this
    function to verify all the indices are in bounds along the
    axis before send it to Tensoflow. If out of bound is detected
    then the caller of this function need to throw
    InvalidArgumentError exception.
    """
        data_shape = tf.cast(tf_shape(data), indices.dtype)
        limit = data_shape[axis]
        cond1 = tf.greater_equal(indices, tf.negative(limit))
        cond2 = tf.less(indices, limit)
        return tf.logical_and(cond1, cond2)

    @classmethod
    def process_neg_idx(cls, data, indices):
        """ Convert all the negative indices to positive
    GatherND and ScatterND/TensorScatterNDUpdate in Tensorflow
    doesn't support negative indices. Therefore need to run this
    function to convert all the negative indices to positive before
    send it to Tensorflow.
    """
        data_shape = tf_shape(data)
        indices_shape = tf_shape(indices)
        max_i = tf.cast(data_shape[: indices_shape[-1]], indices.dtype)
        return tf.math.floormod(tf.add(indices, max_i), max_i)

    @classmethod
    def process_neg_idx_along_axis(cls, data, axis, indices):
        """ Convert all the negative indices to positive
    ScatterND/TensorScatterNDUpdate in Tensorflow doesn't support
    negative indices. Therefore need to run this function to convert
    all the negative indices to positive before send it to Tensorflow.
    """
        data_shape = tf_shape(data)
        max_i = tf.cast(data_shape[axis], indices.dtype)
        return tf.math.floormod(tf.add(indices, max_i), max_i)
