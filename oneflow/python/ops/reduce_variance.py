from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.ops.reduce_mean as reduce_mean
import oneflow.python.ops.math_ops as math_ops


@oneflow_export("math.reduce_variance")
def reduce_std(input_tensor, axis=None, keepdims=False, name=None):
    return math_ops.subtract(
        reduce_mean.reduce_mean(math_ops.square(
            input_tensor), axis, keepdims),
        math_ops.square(reduce_mean.reduce_mean(
            input_tensor, axis, keepdims))
    )
