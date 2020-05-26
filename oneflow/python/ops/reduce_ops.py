from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.ops.user_op_builder as user_op_builder
import oneflow.python.framework.id_util as id_util

import oneflow as flow

@oneflow_export("math.reduce_any")
def reduce_any(x, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceAny_")
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype, name=name+"_constant_scalar"), name+"_not_equal")
    return _get_remote_blob(x, name, "reduce_any", keepdims, axis)

@oneflow_export("math.reduce_min")
def reduce_min(x, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceMin_")
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return x
    return _get_remote_blob(x, name, "reduce_min", keepdims, axis)

@oneflow_export("math.reduce_max")
def reduce_max(x, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceMax_")
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return x
    return _get_remote_blob(x, name, "reduce_max", keepdims, axis)

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
    if axis is None:
        axis = []
    elif isinstance(axis, list) and len(axis) == 0:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype, name=name+"_constant_scalar"), name+"_not_equal")
    return _get_remote_blob(x, name, "reduce_all", keepdims, axis)

@oneflow_export("math.reduce_euclidean_norm")
def reduce_euclidean_norm(input_tensor, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceEuclideanNorm_")
    return flow.math.sqrt(
        flow.math.reduce_sum(
            flow.math.square(input_tensor, name+"_square"),
            axis,
            keepdims,
            name + "_reduce_sum"
        ),
        name + "_sqrt"
    )

@oneflow_export("math.reduce_logsumexp")
def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceLogSumExp_")
    return flow.math.log(
        flow.math.reduce_sum(
            flow.math.exp(input_tensor, name+"_exp"),
            axis,
            keepdims,
            name + "_reduce_sum"
        ),
        name + "_log"
    )

@oneflow_export("math.reduce_std")
def reduce_std(input_tensor, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceStd_")
    if isinstance(axis, list) and len(axis) == 0:
        return flow.zeros_like(input_tensor, dtype=input_tensor.dtype, name=name+"_zeros_like")
    return flow.math.sqrt(
        flow.math.reduce_variance(input_tensor, axis, keepdims, name+"_reduce_variance"),
        name + "_sqrt"
    )

@oneflow_export("math.reduce_variance")
def reduce_variance(input_tensor, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceVariance_")
    if isinstance(axis, list) and len(axis) == 0:
        return flow.zeros_like(input_tensor, dtype=input_tensor.dtype, name=name+"_zeros_like")
    return flow.math.subtract(
        flow.math.reduce_mean(flow.math.square(
            input_tensor, name+"_square_0"), axis, keepdims, name+"_reduce_mean_0"),
        flow.math.square(flow.math.reduce_mean(
            input_tensor, axis, keepdims, name+"_reduce_mean_1"), name+"_square_1"),
        name+"_subtract"
    )


def _get_remote_blob(x, name=None, op_name=None, keepdims=False, axis=None):
    return flow.user_op_builder(name).Op(op_name)\
        .Input("input_tensor", [x])\
        .Output("output_tensor")\
        .Attr("axis", axis, "AttrTypeListInt32")\
        .Attr("keepdims", keepdims, "AttrTypeBool")\
        .Build().InferAndTryRun().RemoteBlobList()[0]

def _check_name(name, unique_name):
    return name if name is not None else id_util.UniqueStr(unique_name)
