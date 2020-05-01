from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.ops.user_op_builder as user_op_builder
import oneflow.python.framework.id_util as id_util

import oneflow as flow

@oneflow_export("math.reduce_any")
def reduce_any(x, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceAny_")
    if len(x.shape) == 1:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype))
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype))
    return _get_remote_blob(x, name, "reduce_any", keepdims, axis)

@oneflow_export("math.reduce_min")
def reduce_min(x, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceMin_")
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return x
    return _get_remote_blob(x, name, "reduce_min", keepdims, axis)

@oneflow_export("math.reduce_prod")
def reduce_prod(x, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceProd_")
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return x
    return _get_remote_blob(x, name, "reduce_prod", keepdims, axis)

@oneflow_export("math.reduce_all")
def reduce_all(x, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceAll_")
    if len(x.shape) == 1:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype))
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype))
    return _get_remote_blob(x, name, "reduce_all", keepdims, axis)

@oneflow_export("math.reduce_euclidean_norm")
def reduce_euclidean_norm(input_tensor, axis=None, keepdims=False):
    return flow.math.sqrt(
        flow.math.reduce_sum(
            flow.math.square(input_tensor),
            axis,
            keepdims
        )
    )

@oneflow_export("math.reduce_logsumexp")
def reduce_logsumexp(input_tensor, axis=None, keepdims=False):
    return flow.math.log(
        flow.math.reduce_sum(
            flow.math.exp(input_tensor),
            axis,
            keepdims
        )
    )

@oneflow_export("math.reduce_std")
def reduce_std(input_tensor, axis=None, keepdims=False):
    return flow.math.sqrt(
        flow.math.reduce_variance(input_tensor, axis, keepdims)
    )

@oneflow_export("math.reduce_variance")
def reduce_variance(input_tensor, axis=None, keepdims=False):
    return flow.math.subtract(
        flow.math.reduce_mean(flow.math.square(
            input_tensor), axis, keepdims),
        flow.math.square(flow.math.reduce_mean(
            input_tensor, axis, keepdims))
    )


def _get_remote_blob(x, name=None, op_name=None, keepdims=False, axis=None):
    return user_op_builder.UserOpConfWrapperBuilder(name).Op(op_name)\
        .Input("input_tensor", [x])\
        .Output("output_tensor")\
        .SetAttr("axis", axis, "AttrTypeListInt32")\
        .SetAttr("keepdims", keepdims, "AttrTypeBool")\
        .Build().RemoteBlobList()[0]

def _check_name(name, unique_name):
    return name if name is not None else id_util.UniqueStr(unique_name)
