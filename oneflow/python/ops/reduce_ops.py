from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.ops.user_math_ops as user_math_ops
import oneflow.python.ops.reduce_sum as reduce_sum
import oneflow.python.ops.math_ops as math_ops
import oneflow.python.ops.user_op_builder as user_op_builder
import oneflow.python.framework.id_util as id_util
import oneflow.python.ops.constant_op as constant_op

def get_remote_blob(x, name=None, op_name=None, keepdims=None, axis=None):
    return user_op_builder.UserOpConfWrapperBuilder(name).Op(op_name)\
        .Input("input_tensor", [x])\
        .Output("output_tensor")\
        .SetAttr("axis", axis, "AttrTypeListInt32")\
        .SetAttr("keepdims", keepdims, "AttrTypeBool")\
        .Build().RemoteBlobList()[0]

def check_name(name, unique_name):
    return name if name is not None else id_util.UniqueStr(unique_name)

@oneflow_export("math.reduce_any")
def reduce_any(x, axis=None, keepdims=None, name=None):
    name = check_name(name, "ReduceAny_")
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return math_ops.not_equal(x, constant_op.constant_scalar(value=0.0, dtype=x.dtype))
    return get_remote_blob(x, name, "reduce_any", keepdims, axis)

@oneflow_export("math.reduce_min")
def reduce_min(x, axis=None, keepdims=None, name=None):
    name = check_name(name, "ReduceMin_")
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return x
    return get_remote_blob(x, name, "reduce_min", keepdims, axis)

@oneflow_export("math.reduce_prod")
def reduce_prod(x, axis=None, keepdims=None, name=None):
    name = check_name(name, "ReduceProd_")
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return x
    return get_remote_blob(x, name, "reduce_prod", keepdims, axis)

@oneflow_export("math.reduce_all")
def reduce_all(x, axis=None, keepdims=None, name=None):
    name = check_name(name, "ReduceAll_")
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return math_ops.not_equal(x, constant_op.constant_scalar(value=0.0, dtype=x.dtype))
    return get_remote_blob(x, name, "reduce_all", keepdims, axis)

@oneflow_export("math.reduce_euclidean_norm")
def reduce_euclidean_norm(input_tensor, axis=None, keepdims=False, name=None):
    return math_ops.sqrt(
        reduce_sum.reduce_sum(
            math_ops.square(input_tensor),
            axis,
            keepdims
        )
    )

@oneflow_export("math.reduce_logsumexp")
def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name=None):
    return user_math_ops.log(
        reduce_sum.reduce_sum(
            user_math_ops.exp(input_tensor),
            axis,
            keepdims
        )
    )

@oneflow_export("math.reduce_std")
def reduce_std(input_tensor, axis=None, keepdims=False, name=None):
    return math_ops.sqrt(
        reduce_variance(input_tensor, axis, keepdims)
    )


@oneflow_export("math.reduce_variance")
def reduce_variance(input_tensor, axis=None, keepdims=False, name=None):
    return math_ops.subtract(
        reduce_mean.reduce_mean(math_ops.square(
            input_tensor), axis, keepdims),
        math_ops.square(reduce_mean.reduce_mean(
            input_tensor, axis, keepdims))
    )
