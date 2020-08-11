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
import os
from typing import Optional, Sequence, Sized, Union

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.remote_blob as remote_blob_util
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
        assert x >= 0 and x < len(shape), "Invalid axis {}, len(shape): {}".format(
            axis, len(shape)
        )

    return axis


def _do_reduce(x, name, op_type_name, keepdims, axis):
    op = (
        flow.user_op_builder(name)
        .Op(op_type_name)
        .Input("input_tensor", [x])
        .Output("output_tensor")
        .Attr("axis", axis)
        .Attr("keepdims", keepdims)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("math.reduce_sum")
def reduce_sum(
    input_tensor: remote_blob_util.BlobDef,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
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

    op = (
        flow.user_op_builder(name)
        .Op("reduce_sum")
        .Input("input_tensor", [input_tensor])
        .Output("output_tensor")
        .Attr("axis", axis)
        .Attr("keepdims", keepdims)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("math.reduce_any")
def reduce_any(
    x: remote_blob_util.BlobDef,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    name = _gen_unique_name_if_need(name, "ReduceAny_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype))
    return _do_reduce(x, name, "reduce_any", keepdims, axis)


@oneflow_export("math.reduce_min")
def reduce_min(
    x: remote_blob_util.BlobDef,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    name = _gen_unique_name_if_need(name, "ReduceMin_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return x
    return _do_reduce(x, name, "reduce_min", keepdims, axis)


@oneflow_export("math.reduce_max")
def reduce_max(
    x: remote_blob_util.BlobDef,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    name = _gen_unique_name_if_need(name, "ReduceMax_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return x
    return _do_reduce(x, name, "reduce_max", keepdims, axis)


@oneflow_export("math.reduce_prod")
def reduce_prod(
    x: remote_blob_util.BlobDef,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    name = _gen_unique_name_if_need(name, "ReduceProd_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return x
    return _do_reduce(x, name, "reduce_prod", keepdims, axis)


@oneflow_export("math.reduce_all")
def reduce_all(
    x: remote_blob_util.BlobDef,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    name = _gen_unique_name_if_need(name, "ReduceAll_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype))
    return _do_reduce(x, name, "reduce_all", keepdims, axis)


@oneflow_export("math.reduce_euclidean_norm")
def reduce_euclidean_norm(
    input_tensor: remote_blob_util.BlobDef,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    name = _gen_unique_name_if_need(name, "ReduceEuclideanNorm_")
    return flow.math.sqrt(
        flow.math.reduce_sum(
            flow.math.square(input_tensor, name + "_square"),
            axis,
            keepdims,
            name + "_reduce_sum",
        ),
        name + "_sqrt",
    )


@oneflow_export("math.reduce_logsumexp")
def reduce_logsumexp(
    input_tensor: remote_blob_util.BlobDef,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    name = _gen_unique_name_if_need(name, "ReduceLogSumExp_")
    axis = _check_axis(axis, input_tensor.shape)
    return flow.math.log(
        flow.math.reduce_sum(
            flow.math.exp(input_tensor, name + "_exp"),
            axis,
            keepdims,
            name + "_reduce_sum",
        ),
        name + "_log",
    )


@oneflow_export("math.reduce_std")
def reduce_std(
    input_tensor: remote_blob_util.BlobDef,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    name = _gen_unique_name_if_need(name, "ReduceStd_")
    axis = _check_axis(axis, input_tensor.shape)
    if isinstance(axis, list) and len(axis) == 0:
        return flow.zeros_like(
            input_tensor, dtype=input_tensor.dtype, name=name + "_zeros_like"
        )
    return flow.math.sqrt(
        flow.math.reduce_variance(
            input_tensor, axis, keepdims, name + "_reduce_variance"
        ),
        name + "_sqrt",
    )


@oneflow_export("math.reduce_variance")
def reduce_variance(
    input_tensor: remote_blob_util.BlobDef,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    name = _gen_unique_name_if_need(name, "ReduceVariance_")
    axis = _check_axis(axis, input_tensor.shape)
    if isinstance(axis, list) and len(axis) == 0:
        return flow.zeros_like(
            input_tensor, dtype=input_tensor.dtype, name=name + "_zeros_like"
        )
    return flow.math.subtract(
        flow.math.reduce_mean(
            flow.math.square(input_tensor, name + "_square_minuend"),
            axis,
            keepdims,
            name + "_reduce_mean_minuend",
        ),
        flow.math.square(
            flow.math.reduce_mean(
                input_tensor, axis, keepdims, name + "_reduce_mean_subtrahend"
            ),
            name + "_square_subtrahend",
        ),
        name + "_subtract",
    )
