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
from __future__ import absolute_import

import os
from typing import Union, Optional, Sequence

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.framework.module as module_util
import oneflow.python.ops.math_unary_elementwise_ops as math_unary_elementwise_ops
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("math.add")
def add(
    x: Union[int, float, remote_blob_util.BlobDef],
    y: Union[int, float, remote_blob_util.BlobDef],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if isinstance(x, (int, float)):
        return scalar_add(y, x, name)
    elif isinstance(y, (int, float)):
        return scalar_add(x, y, name)
    elif x.shape == y.shape and x.batch_axis == y.batch_axis:
        return element_wise_add(x, y, name)
    elif x.shape == (1,):
        return scalar_add_by_tensor(y, x, name)
    elif y.shape == (1,):
        return scalar_add_by_tensor(x, y, name)
    else:
        return broadcast_add(x, y, name)


def _recursive_build_add_n(inputs, name=None):
    inputs = list(inputs)
    kernel_max_inputs = 8
    if len(inputs) == 1:
        return inputs[0]
    elif len(inputs) <= kernel_max_inputs:
        return (
            flow.user_op_builder(
                name if name is not None else id_util.UniqueStr("AddN_")
            )
            .Op("add_n")
            .Input("in", inputs)
            .Output("out")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
        assert len(inputs) > kernel_max_inputs
        new_inputs = inputs[kernel_max_inputs:]
        new_inputs.append(_recursive_build_add_n(inputs[:kernel_max_inputs]))
        return _recursive_build_add_n(new_inputs)


@oneflow_export("math.add_n")
def add_n(
    inputs: Sequence[remote_blob_util.BlobDef], name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    return _recursive_build_add_n(inputs, name)


@oneflow_export("math.subtract")
def subtract(
    x: Union[int, float, remote_blob_util.BlobDef],
    y: Union[int, float, remote_blob_util.BlobDef],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if isinstance(x, (int, float)):
        return scalar_add(-1 * y, x, name)
    elif isinstance(y, (int, float)):
        return scalar_add(x, -1 * y, name)
    elif x.shape == y.shape:
        # TODO: add element-wise op
        return broadcast_sub(x, y, name)
    elif x.shape == (1,):
        return scalar_sub_by_tensor(y, x, name)
    elif y.shape == (1,):
        return scalar_sub_by_tensor(x, y, name)
    else:
        return broadcast_sub(x, y, name)


@oneflow_export("math.multiply")
def multiply(
    x: Union[int, float, remote_blob_util.BlobDef],
    y: Union[int, float, remote_blob_util.BlobDef],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if isinstance(x, (int, float)):
        return scalar_mul(y, x, name)
    elif isinstance(y, (int, float)):
        return scalar_mul(x, y, name)
    elif x.shape == y.shape and x.batch_axis == y.batch_axis:
        return element_wise_mul(x, y, name)
    elif x.shape == (1,):
        return scalar_mul_by_tensor(y, x, name)
    elif y.shape == (1,):
        return scalar_mul_by_tensor(x, y, name)
    else:
        return broadcast_mul(x, y, name)


@oneflow_export("math.divide")
def divide(
    x: Union[int, float, remote_blob_util.BlobDef],
    y: Union[int, float, remote_blob_util.BlobDef],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if isinstance(x, (int, float)):
        return scalar_mul(math_unary_elementwise_ops.reciprocal_no_nan(y), x, name)
    elif isinstance(y, (int, float)):
        if y == 0 or y == 0.0:
            y = 0.0
        else:
            y = 1.0 / (float(y))
        return scalar_mul(x, y, name)
    elif x.shape == y.shape:
        # TODO: add element-wise op
        return broadcast_div(x, y, name)
    elif x.shape == (1,):
        return scalar_div_by_tensor(y, x, name)
    elif y.shape == (1,):
        return scalar_div_by_tensor(x, y, name)
    else:
        return broadcast_div(x, y, name)


@oneflow_export("math.mod")
def floor_mod(
    x: Union[int, float, remote_blob_util.BlobDef],
    y: Union[int, float, remote_blob_util.BlobDef],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if isinstance(x, (int, float)):
        raise NotImplementedError
    elif isinstance(y, (int, float)):
        raise NotImplementedError
    elif x.shape == y.shape:
        # TODO: add element-wise op
        return broadcast_floor_mod(x, y, name)
    else:
        return broadcast_floor_mod(x, y, name)


def scalar_add(x, operand, name=None):
    if name is None:
        name = id_util.UniqueStr("ScalarAdd_")
    builder = flow.user_op_builder(name).Op("scalar_add").Input("in", [x]).Output("out")
    if isinstance(operand, int):
        builder = (
            builder.Attr("has_int_operand", True)
            .Attr("has_float_operand", False)
            .Attr("int_operand", operand)
            .Attr("float_operand", 0.0)
        )
    elif isinstance(operand, float):
        builder = (
            builder.Attr("has_int_operand", False)
            .Attr("has_float_operand", True)
            .Attr("int_operand", 0)
            .Attr("float_operand", operand)
        )
    return builder.Build().InferAndTryRun().RemoteBlobList()[0]


def scalar_add_by_tensor(x, scalar, name=None):
    return (
        flow.user_op_builder(name or id_util.UniqueStr("ScalarAddByTensor_"))
        .Op("scalar_add_by_tensor")
        .Input("x", [x])
        .Input("scalar", [scalar])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def element_wise_add(x, y, name=None):
    return flow.math.add_n([x, y], name)


def build_broadcast_binary_op(math_op, x, y, name=None):
    if name is None:
        name = id_util.UniqueStr(math_op + "_")
    return (
        flow.user_op_builder(name)
        .Op(math_op)
        .Input("x", [x])
        .Input("y", [y])
        .Output("z")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def broadcast_add(x, y, name=None):
    return build_broadcast_binary_op("broadcast_add", x, y, name)


def broadcast_sub(x, y, name=None):
    return build_broadcast_binary_op("broadcast_sub", x, y, name)


def scalar_sub_by_tensor(x, scalar, name=None):
    return (
        flow.user_op_builder(name or id_util.UniqueStr("ScalarSubByTensor_"))
        .Op("scalar_sub_by_tensor")
        .Input("x", [x])
        .Input("scalar", [scalar])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def element_wise_mul(x, y, name=None):
    return (
        flow.user_op_builder(name or id_util.UniqueStr("ElementWiseMul_"))
        .Op("multiply")
        .Input("x", [x])
        .Input("y", [y])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def broadcast_mul(x, y, name=None):
    return build_broadcast_binary_op("broadcast_mul", x, y, name)


def scalar_mul(x, operand, name=None):
    if name is None:
        name = id_util.UniqueStr("ScalarMul_")
    builder = flow.user_op_builder(name).Op("scalar_mul").Input("in", [x]).Output("out")
    if isinstance(operand, int):
        builder = (
            builder.Attr("has_int_operand", True)
            .Attr("has_float_operand", False)
            .Attr("int_operand", operand)
            .Attr("float_operand", 0.0)
        )
    elif isinstance(operand, float):
        builder = (
            builder.Attr("has_int_operand", False)
            .Attr("has_float_operand", True)
            .Attr("int_operand", 0)
            .Attr("float_operand", operand)
        )
    return builder.Build().InferAndTryRun().RemoteBlobList()[0]


def scalar_mul_by_tensor(x, scalar, name=None):
    return (
        flow.user_op_builder(name or id_util.UniqueStr("ScalarMulByTensor_"))
        .Op("scalar_mul_by_tensor")
        .Input("x", [x])
        .Input("scalar", [scalar])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def broadcast_div(x, y, name=None):
    return build_broadcast_binary_op("broadcast_div", x, y, name)


def scalar_div_by_tensor(x, scalar, name=None):
    return (
        flow.user_op_builder(name or id_util.UniqueStr("ScalarDivByTensor_"))
        .Op("scalar_div_by_tensor")
        .Input("x", [x])
        .Input("scalar", [scalar])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def broadcast_floor_mod(x, y, name=None):
    return build_broadcast_binary_op("broadcast_floor_mod", x, y, name)


@oneflow_export("math.tanh")
def tanh(
    x: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Computes hyperbolic tangent of `x` element-wise.

    Args:
        x: Input `Blob`.
    Returns:
        A `Blob`
    """

    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("TanH_"))
        .Op("tanh")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.gelu")
def gelu(
    x: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Gaussian Error Linear Units.

    Args:
        x: Input `Blob`.
    Returns:
        A `Blob`
    """
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Gelu_"))
        .Op("gelu")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.relu", "nn.relu")
def relu(
    x: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""ReLU activation

    Args:
        x: Input `Blob`.
    Returns:
        A `Blob`
    """

    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Relu_"))
        .Op("relu")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.sigmoid")
def sigmoid(
    x: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Computes sigmoid of `x` element-wise.

    Args:
        x: Input `Blob`.
    Returns:
        A `Blob`
    """
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Sigmoid_")
        )
        .Op("sigmoid")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.unsorted_segment_sum", "unsorted_segment_sum")
def unsorted_segment_sum(
    data: remote_blob_util.BlobDef,
    segment_ids: remote_blob_util.BlobDef,
    num_segments: int,
    axis: int = 0,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("UnsortedSegmentSum_")
        )
        .Op("unsorted_segment_sum")
        .Input("data", [data])
        .Input("segment_ids", [segment_ids])
        .Output("out")
        .Attr("axis", int(axis))
        .Attr("num_segments", int(num_segments))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.unsorted_segment_sum_like", "unsorted_segment_sum_like")
def unsorted_segment_sum_like(
    data: remote_blob_util.BlobDef,
    segment_ids: remote_blob_util.BlobDef,
    like: remote_blob_util.BlobDef,
    axis: int = 0,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("UnsortedSegmentSumLike_")
        )
        .Op("unsorted_segment_sum_like")
        .Input("data", [data])
        .Input("segment_ids", [segment_ids])
        .Input("like", [like])
        .Output("out")
        .Attr("axis", int(axis))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.unsorted_batch_segment_sum", "unsorted_batch_segment_sum")
def unsorted_batch_segment_sum(
    data: remote_blob_util.BlobDef,
    segment_ids: remote_blob_util.BlobDef,
    num_segments: int,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("UnsortedBatchSegmentSum_")
        )
        .Op("unsorted_batch_segment_sum")
        .Input("data", [data])
        .Input("segment_ids", [segment_ids])
        .Output("out")
        .Attr("num_segments", int(num_segments))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("cast")
def cast(
    x: remote_blob_util.BlobDef, dtype: dtype_util.dtype, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Return a `Blob` of given data type `dtype` and indentical shape to `x`

    Args:
        x: a `Blob`.
        dtype: a OneFlow data type. For instance, `oneflow.float`.
    Returns:
        A `Blob`
    """
    if x.dtype == dtype:
        return x
    if name is None:
        name = id_util.UniqueStr("Cast_")

    return (
        flow.user_op_builder(name)
        .Op("cast")
        .Input("in", [x])
        .Output("out")
        .Attr("dtype", dtype)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.equal")
def equal(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    return build_broadcast_binary_op("broadcast_equal", x, y, name)


@oneflow_export("math.not_equal")
def not_equal(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    return build_broadcast_binary_op("broadcast_not_equal", x, y, name)


@oneflow_export("math.less")
def less(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    return build_broadcast_binary_op("broadcast_less", x, y, name)


@oneflow_export("math.less_equal")
def less_equal(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    return build_broadcast_binary_op("broadcast_less_equal", x, y, name)


@oneflow_export("math.greater")
def greater(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    return build_broadcast_binary_op("broadcast_greater", x, y, name)


@oneflow_export("math.greater_equal")
def greater_equal(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    return build_broadcast_binary_op("broadcast_greater_equal", x, y, name)


@oneflow_export("math.logical_and")
def logical_and(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    return build_broadcast_binary_op("broadcast_logical_and", x, y, name)


@oneflow_export("math.minimum")
def broadcast_min(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    return build_broadcast_binary_op("broadcast_minimum", x, y, name)


@oneflow_export("math.maximum")
def broadcast_max(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    return build_broadcast_binary_op("broadcast_maximum", x, y, name)


@oneflow_export("math.reduced_shape_elem_cnt")
def elem_cnt(
    input_blob: remote_blob_util.BlobDef,
    axis: Optional[Sequence[int]] = None,
    dtype: Optional[dtype_util.dtype] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ShapeElemCnt_"),
    )
    op_conf.shape_elem_cnt_conf.x = input_blob.unique_name
    if axis is None:
        op_conf.shape_elem_cnt_conf.exclude_axis_conf.SetInParent()
    else:
        assert isinstance(axis, (tuple, list))
        op_conf.shape_elem_cnt_conf.include_axis_conf.axis.extend(axis)
    if dtype is not None:
        op_conf.shape_elem_cnt_conf.data_type = dtype.oneflow_proto_dtype
    op_conf.shape_elem_cnt_conf.y = "y"
    interpret_util.Forward(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    out_lbi.op_name = op_conf.name
    out_lbi.blob_name = "y"
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("math.top_k")
def top_k(
    input: remote_blob_util.BlobDef,
    k: int = 1,
    sorted: bool = True,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("TopK_"))
        .Op("top_k")
        .Input("in", [input])
        .Output("out")
        .Attr("k", k)
        .Attr("sorted", sorted)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.argmax")
def argmax(
    input: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("ArgMax_"))
        .Op("argmax")
        .Input("in", [input])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.broadcast_to_compatible_with", "broadcast_to_compatible_with")
def broadcast_to_compatible_with(
    x: remote_blob_util.BlobDef,
    compatible: Sequence[remote_blob_util.BlobDef],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    assert isinstance(compatible, (list, tuple))
    if name is None:
        name = id_util.UniqueStr("BroadcastToCompatibleWith_")

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.broadcast_to_compatible_with_conf, "x", x.unique_name)
    setattr(op_conf.broadcast_to_compatible_with_conf, "y", "y")
    op_conf.broadcast_to_compatible_with_conf.compatible.extend(
        [cp.unique_name for cp in compatible]
    )
    interpret_util.Forward(op_conf)

    ret_lbi = logical_blob_id_util.LogicalBlobId()
    ret_lbi.op_name = op_conf.name
    ret_lbi.blob_name = "y"
    return remote_blob_util.RemoteBlob(ret_lbi)


@oneflow_export(
    "math.clip_by_value", "clip_by_value", "clip_by_scalar", "clip", "clamp"
)
def clip_by_value(
    values: remote_blob_util.BlobDef,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if name is None:
        name = id_util.UniqueStr("ClipByValue_")

    if min_value is not None and max_value is not None:
        op_builder = (
            flow.user_op_builder(name)
            .Op("clip_by_scalar")
            .Attr("floating_min", float(min_value))
            .Attr("integral_min", int(min_value))
            .Attr("floating_max", float(max_value))
            .Attr("integral_max", int(max_value))
        )
    elif min_value is not None:
        op_builder = (
            flow.user_op_builder(name)
            .Op("clip_by_scalar_min")
            .Attr("floating_min", float(min_value))
            .Attr("integral_min", int(min_value))
        )
    elif max_value is not None:
        op_builder = (
            flow.user_op_builder(name)
            .Op("clip_by_scalar_max")
            .Attr("floating_max", float(max_value))
            .Attr("integral_max", int(max_value))
        )
    else:
        raise ValueError("min_value and max_value cannot be None at the same time")

    return (
        op_builder.Input("x", [values])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.l2_normalize")
def l2_normalize(
    input: remote_blob_util.BlobDef,
    axis: Optional[int] = None,
    epsilon: float = 1e-12,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if axis < 0:
        axis += len(input.shape)
    assert axis >= 0 and axis < len(input.shape)
    y, square_x_sum = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("L2Normalize_")
        )
        .Op("l2_normalize")
        .Input("x", [input])
        .Output("y")
        .Output("square_x_sum")
        .Attr("axis", int(axis))
        .Attr("epsilon", float(epsilon))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return y


@oneflow_export("math.squared_difference")
def squared_difference(
    x: Union[int, float, remote_blob_util.BlobDef],
    y: Union[int, float, remote_blob_util.BlobDef],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    name_subtract, name_square = None, None
    if name is not None:
        name_subtract = name + "_subtract"
        name_square = name + "_square"
    return flow.math.square(flow.math.subtract(x, y, name_subtract), name_square)


@oneflow_export("math.gelu_grad")
def gelu_grad(
    x: remote_blob_util.BlobDef,
    dy: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("GeluGrad_")
        )
        .Op("gelu_grad")
        .Input("x", [x])
        .Input("dy", [dy])
        .Output("dx")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.tanh_grad")
def tanh_grad(
    y: remote_blob_util.BlobDef,
    dy: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("TanhGrad_")
        )
        .Op("tanh_grad")
        .Input("y", [y])
        .Input("dy", [dy])
        .Output("dx")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
