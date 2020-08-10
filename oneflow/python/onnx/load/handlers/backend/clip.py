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


@onnx_op("Clip")
class Clip(BackendHandler):
    @classmethod
    def args_check(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        # uint64 cannot upcast to any tensorflow supported datatype
        # for tf.clip_by_value that didn't lose precision
        if x.dtype == tf.uint64:
            exception.OP_UNSUPPORTED_EXCEPT(
                "Clip input, min and max in " + str(x.dtype) + " datatype", "Tensorflow"
            )

    @classmethod
    def _common(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]
        x_dtype = x.dtype

        if cls.SINCE_VERSION < 11:
            # min/max were required and passed as attributes
            clip_value_min = node.attrs.get("min", tf.reduce_min(x))
            clip_value_max = node.attrs.get("max", tf.reduce_max(x))
        else:
            # min/max are optional and passed as inputs
            clip_value_min = (
                tensor_dict[node.inputs[1]] if len(node.inputs) > 1 else x_dtype.min
            )
            clip_value_max = (
                tensor_dict[node.inputs[2]] if len(node.inputs) > 2 else x_dtype.max
            )

        # tf.clip_by_value doesn't support uint8, uint16, uint32, int8 and int16
        # dtype for x, therefore need to upcast it to tf.int32 or tf.int64
        if x_dtype in [tf.uint8, tf.uint16, tf.uint32, tf.int8, tf.int16]:
            cast_to = tf.int64 if x_dtype == tf.uint32 else tf.int32
            x = tf.cast(x, cast_to)
            clip_value_min = tf.cast(clip_value_min, cast_to)
            clip_value_max = tf.cast(clip_value_max, cast_to)
            y = tf.clip_by_value(x, clip_value_min, clip_value_max)
            y = tf.cast(y, x_dtype)
        else:
            y = tf.clip_by_value(x, clip_value_min, clip_value_max)

        return [y]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        return cls._common(node, **kwargs)
