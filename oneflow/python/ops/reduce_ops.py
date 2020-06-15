import oneflow as flow
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import os

from oneflow.python.oneflow_export import oneflow_export


def _gen_unique_name_if_need(name, default_name):
    if name is None:
        return id_util.UniqueStr(default_name)

    assert isinstance(name, str), name
    return name


def _check_axis(axis, shape):
    if axis is None:
        axis = list(range(len(shape)))

    if isinstance(axis, int):
        axis = [axis]

    assert isinstance(axis, (list, tuple)), "Invalid axis {}".format(axis)
    for x in axis:
        if x < 0:
            x += len(shape)
        assert x >= 0 and x < len(shape), "Invalid axis {}".format(axis)

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


@oneflow_export("math.reduce_sum")
def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
    r"""Computes the sum of elements across dimensions of a tensor.

    Args:
        input_tensor: A `Blob`.
        axis: The dimensions to reduce. If None (by default), reduces all dimensions.
            Must be a int or a list/tuple of int which must be in the range
            [-len(input_tensor.shape), len(input_tensor.shape))
        keepdims: If true, reduced dimensions will be kept with length 1.
        name: A name for the operator.
    Returns:
        A `Blob`.
    """
    name = _gen_unique_name_if_need(name, "ReduceSum_")

    axis = _check_axis(axis, input_tensor.shape)
    if len(axis) == 0:
        return input_tensor

    if os.getenv("ENABLE_USER_OP") == "True":
        op = (
            flow.user_op_builder(name)
            .Op("reduce_sum")
            .Input("input_tensor", [input_tensor])
            .Output("output_tensor")
            .Attr("axis", axis, "AttrTypeListInt32")
            .Attr("keepdims", keepdims, "AttrTypeBool")
            .Build()
        )
        return op.InferAndTryRun().SoleOutputBlob()
    else:
        op_conf = op_conf_util.OperatorConf()
        setattr(op_conf, "name", name)
        setattr(op_conf.reduce_sum_conf, "in", input_tensor.unique_name)
        setattr(op_conf.reduce_sum_conf, "out", "out")
        op_conf.reduce_sum_conf.axis[:] = list(axis)
        setattr(op_conf.reduce_sum_conf, "keep_dims", keepdims)
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.reduce_any")
def reduce_any(x, axis=None, keepdims=False, name=None):
    name = _gen_unique_name_if_need(name, "ReduceAny_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype))
    return _do_reduce(x, name, "reduce_any", keepdims, axis)


@oneflow_export("math.reduce_min")
def reduce_min(x, axis=None, keepdims=False, name=None):
    name = _gen_unique_name_if_need(name, "ReduceMin_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return x
    return _do_reduce(x, name, "reduce_min", keepdims, axis)


@oneflow_export("math.reduce_prod")
def reduce_prod(x, axis=None, keepdims=False, name=None):
    name = _gen_unique_name_if_need(name, "ReduceProd_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return x
    return _do_reduce(x, name, "reduce_prod", keepdims, axis)


@oneflow_export("math.reduce_all")
def reduce_all(x, axis=None, keepdims=False, name=None):
    name = _gen_unique_name_if_need(name, "ReduceAll_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype))
    return _do_reduce(x, name, "reduce_all", keepdims, axis)


@oneflow_export("math.reduce_euclidean_norm")
def reduce_euclidean_norm(input_tensor, axis=None, keepdims=False, name=None):
    return flow.math.sqrt(
        flow.math.reduce_sum(flow.math.square(input_tensor), axis, keepdims)
    )


@oneflow_export("math.reduce_logsumexp")
def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name=None):
    return flow.math.log(
        flow.math.reduce_sum(flow.math.exp(input_tensor), axis, keepdims)
    )


@oneflow_export("math.reduce_std")
def reduce_std(input_tensor, axis=None, keepdims=False, name=None):
    return flow.math.sqrt(flow.math.reduce_variance(input_tensor, axis, keepdims))


@oneflow_export("math.reduce_variance")
def reduce_variance(input_tensor, axis=None, keepdims=False, name=None):
    return flow.math.subtract(
        flow.math.reduce_mean(flow.math.square(input_tensor), axis, keepdims),
        flow.math.square(flow.math.reduce_mean(input_tensor, axis, keepdims)),
    )
