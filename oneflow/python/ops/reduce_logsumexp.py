from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.ops.user_math_ops as user_math_ops
import oneflow.python.ops.reduce_sum as reduce_sum


@oneflow_export("math.reduce_logsumexp")
def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name=None):
    return user_math_ops.log(
        reduce_sum.reduce_sum(
            user_math_ops.exp(input_tensor),
            axis,
            keepdims
        )
    )
