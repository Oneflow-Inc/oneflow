from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.ops.user_math_ops as user_math_ops
import oneflow.python.ops.reduce_sum as reduce_sum
import oneflow.python.ops.math_ops as math_ops
import oneflow.python.ops.user_op_builder as user_op_builder
import oneflow.python.framework.id_util as id_util
import oneflow.python.ops.constant_op as constant_op

@oneflow_export("math.reduce_any")
def reduce_any(x, axis=None, keepdims=None, name=None):
    if name is None:
        name = id_util.UniqueStr("ReduceAny_")
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return math_ops.not_equal(x, constant_op.constant_scalar(value=0.0, dtype=x.dtype))
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("reduce")\
        .Input("tensor_in", [x])\
        .Output("tensor_out")\
        .SetAttr("axis", axis, "AttrTypeListInt32")\
        .SetAttr("keepdims", keepdims, "AttrTypeBool")\
        .SetAttr("reduce_func_type", "Any", "AttrTypeString")\
        .Build().RemoteBlobList()[0]

@oneflow_export("math.reduce_min")
def reduce_min(x, axis=None, keepdims=None, name=None):
    if name is None:
        name = id_util.UniqueStr("ReduceMin_")
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return x
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("reduce")\
        .Input("tensor_in", [x])\
        .Output("tensor_out")\
        .SetAttr("axis", axis, "AttrTypeListInt32")\
        .SetAttr("keepdims", keepdims, "AttrTypeBool")\
        .SetAttr("reduce_func_type", "Min", "AttrTypeString")\
        .Build().RemoteBlobList()[0]

@oneflow_export("math.reduce_prod")
def reduce_prod(x, axis=None, keepdims=None, name=None):
    if name is None:
        name = id_util.UniqueStr("ReduceProd_")
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return x
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("reduce")\
        .Input("tensor_in", [x])\
        .Output("tensor_out")\
        .SetAttr("axis", axis, "AttrTypeListInt32")\
        .SetAttr("keepdims", keepdims, "AttrTypeBool")\
        .SetAttr("reduce_func_type", "Prod", "AttrTypeString")\
        .Build().RemoteBlobList()[0]


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
        math_ops.subtract(
            reduce_mean.reduce_mean(math_ops.square(
                input_tensor), axis, keepdims),
            math_ops.square(reduce_mean.reduce_mean(
                input_tensor, axis, keepdims))
        )
    )


@oneflow_export("math.reduce_variance")
def reduce_std(input_tensor, axis=None, keepdims=False, name=None):
    return math_ops.subtract(
        reduce_mean.reduce_mean(math_ops.square(
            input_tensor), axis, keepdims),
        math_ops.square(reduce_mean.reduce_mean(
            input_tensor, axis, keepdims))
    )
