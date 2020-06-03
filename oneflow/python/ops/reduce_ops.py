import oneflow as flow
import oneflow.python.framework.id_util as id_util

from oneflow.python.oneflow_export import oneflow_export


def _check_name(name, unique_name):
    if name is None:
        return id_util.UniqueStr(unique_name)
    return name


def _check_axis(axis, shape):
    if axis is None:
        axis = list(range(len(shape)))

    if isinstance(axis, int):
        axis = [axis]

    assert isinstance(axis, (list, tuple)), "Invalid axis {}".format(axis)
    return axis


def _do_reduce(x, name, op_type_name, keepdims, axis):
    op = (
        flow.user_op_builder(name)
        .Op(op_type_name)
        .Input("input_tensor", [x])
        .Output("output_tensor")
        .Attr("axis", axis, "AttrTypeListInt32")
        .Attr("keepdims", keepdims, "AttrTypeBool")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("math.reduce_any")
def reduce_any(x, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceAny_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype))
    return _do_reduce(x, name, "reduce_any", keepdims, axis)


@oneflow_export("math.reduce_min")
def reduce_min(x, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceMin_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return x
    return _do_reduce(x, name, "reduce_min", keepdims, axis)


@oneflow_export("math.reduce_prod")
def reduce_prod(x, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceProd_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return x
    return _do_reduce(x, name, "reduce_prod", keepdims, axis)


@oneflow_export("math.reduce_all")
def reduce_all(x, axis=None, keepdims=False, name=None):
    name = _check_name(name, "ReduceAll_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype))
    return _do_reduce(x, name, "reduce_all", keepdims, axis)


@oneflow_export("math.reduce_euclidean_norm")
def reduce_euclidean_norm(input_tensor, axis=None, keepdims=False, name=None):
    return flow.math.sqrt(flow.math.reduce_sum(flow.math.square(input_tensor), axis, keepdims))


@oneflow_export("math.reduce_logsumexp")
def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name=None):
    return flow.math.log(flow.math.reduce_sum(flow.math.exp(input_tensor), axis, keepdims))


@oneflow_export("math.reduce_std")
def reduce_std(input_tensor, axis=None, keepdims=False, name=None):
    return flow.math.sqrt(flow.math.reduce_variance(input_tensor, axis, keepdims))


@oneflow_export("math.reduce_variance")
def reduce_variance(input_tensor, axis=None, keepdims=False, name=None):
    return flow.math.subtract(
        flow.math.reduce_mean(flow.math.square(input_tensor), axis, keepdims),
        flow.math.square(flow.math.reduce_mean(input_tensor, axis, keepdims)),
    )
