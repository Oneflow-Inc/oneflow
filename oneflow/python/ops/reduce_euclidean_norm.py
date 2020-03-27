from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.ops.user_math_ops as user_math_ops
import oneflow.python.ops.reduce_sum as reduce_sum
import oneflow.python.ops.math_ops as math_ops


@oneflow_export("math.reduce_euclidean_norm")
def reduce_euclidean_norm(input_tensor, axis=None, keepdims=False, name=None):
    return math_ops.sqrt(
        reduce_sum.reduce_sum(
            math_ops.square(input_tensor),
            axis,
            keepdims
        )
    )
