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

from oneflow.python.onnx.load.common import get_data_format
from oneflow.python.onnx.load.common import get_perm_from_formats
from oneflow.python.onnx.load.common.tf_helper import tf_shape


class UnpoolMixin(object):
    @classmethod
    def max_unpool(cls, node, input_dict):
        """
            MaxUnpooling operation
        """
        x = input_dict[node.inputs[0]]
        ind = input_dict[node.inputs[1]]
        if len(node.inputs) > 2:
            output_shape = input_dict.get(node.inputs[2], None)
        else:
            output_shape = None

        kernel_shape = node.attrs["kernel_shape"]

        spatial_size = len(kernel_shape)
        x_rank = spatial_size + 2
        storage_format, _ = get_data_format(x_rank)

        # if strides are not provided default is 1 along each spatial axis
        strides = node.attrs.get("strides", [1] * spatial_size)
        pads = node.attrs.get("pads", None)

        input_shape = tf_shape(x)
        default_shape = cls._get_default_shape(input_shape, kernel_shape, strides)

        need_trans = storage_format != "NHWC"
        if need_trans:
            x = tf.transpose(x, perm=get_perm_from_formats(storage_format, "NHWC"))
            ind = tf.transpose(ind, perm=get_perm_from_formats(storage_format, "NHWC"))

        # default_shape to NHWC storage format
        default_shape = [input_shape[0]] + default_shape + [input_shape[1]]

        unpooled = cls._unpool(x, ind, default_shape)

        if need_trans:
            unpooled = tf.transpose(
                unpooled, perm=get_perm_from_formats("NHWC", storage_format)
            )

        if output_shape is not None:
            pads = cls._get_pads_from_output_shape(unpooled, output_shape)
        if pads is not None:
            unpooled = cls._pad_output(unpooled, pads, 0)

        return [unpooled]

    @classmethod
    def _get_default_shape(cls, input_shape, kernel_shape, strides):
        """
            Calculates default shape from kernel_shape and strides
            Args:
                input_shape:   shape of the input to unpool op
                kernel_shape:  the size of the kernel along each axis
                output_shape:  stride along each spatial axis
          Return:
            default_shape: calculated default_shape
        """
        default_shape = []
        for d in range(len(kernel_shape)):
            default_shape.append(
                (input_shape[d + 2] - 1) * int(strides[d]) + int(kernel_shape[d])
            )
        return default_shape

    @classmethod
    def _get_pads_from_output_shape(cls, unpool, output_shape):
        """
            Calculates the paddings from specified output_shape
            Args:
                unpool:       result from unpool operation
                output_shape: expected shape of the output
            Return:
                pads:         calculated paddings in format
                              [x1_begin, x2_begin,.., x1_end, x2_end]
                              where xi_... represent pads added to begin
                              or end of axis i
        """
        unpool_shape = tf.cast(tf.shape(unpool), dtype=tf.int32)
        new_shape = tf.cast(output_shape, dtype=tf.int32)

        pads_begin = []
        pads_end = []

        for d in range(len(unpool.get_shape())):
            pad_total = new_shape[d] - unpool_shape[d]
            pad_begin = tf.cast(pad_total / 2, tf.int32)
            pad_end = pad_total - pad_begin
            pads_begin = pads_begin + [pad_begin]
            pads_end = pads_end + [pad_end]

        pads = pads_begin + pads_end
        return pads

    @classmethod
    def _pad_output(cls, unpool, pads, constant_values):
        """
            Pad the output from unpool op
            Args:
                unpool:         result from unpool op
                pads:           paddings in format
                                [x1_begin, x2_begin,..., x1_end, x2_end]
                constant_values: constant value to fill up the padded spaces
            Return:
                padded:         padded tensor
        """
        unpool_shape = unpool.get_shape()
        paddings = []
        for d in range(len(unpool_shape)):
            paddings = paddings + [[pads[d], pads[d + len(unpool_shape)]]]
        padded = tf.pad(unpool, paddings, "CONSTANT", constant_values=constant_values)
        return padded

    @classmethod
    def _unpool(cls, pool, ind, output_shape, scope="unpool"):
        """
            Unpooling layer after max_pool_with_argmax.

            Args:
                pool:          max pooled output tensor
                ind:           argmax indices
                output_shape:  the shape of the output
            Return:
                unpool:        unpooling tensor
        """
        with tf.compat.v1.variable_scope(scope):
            input_shape = tf.shape(pool)

            flat_input_size = tf.reduce_prod(input_shape)
            flat_output_shape = [
                output_shape[0],
                output_shape[1] * output_shape[2] * output_shape[3],
            ]

            pool_ = tf.reshape(pool, [flat_input_size])
            batch_range = tf.reshape(
                tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                shape=[input_shape[0], 1, 1, 1],
            )
            b = tf.ones_like(ind) * batch_range
            b1 = tf.reshape(b, [flat_input_size, 1])
            ind_ = tf.reshape(ind, [flat_input_size, 1])
            ind_ = tf.concat([b1, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
            ret = tf.reshape(ret, output_shape)
        return ret
