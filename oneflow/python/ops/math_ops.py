from __future__ import absolute_import

import os
from typing import Union, Optional, Sequence

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.ops.math_unary_elementwise_ops as math_unary_elementwise_ops
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("math.add")
def add(
    x: Union[int, float, remote_blob_util.BlobDef],
    y: Union[int, float, remote_blob_util.BlobDef],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Computes 'x' add 'y'
 
    Args:
        x: A 'Blob' with type of 'int' or 'float' 
        y: A 'Blob' with type of 'int' or 'float' 
        name: This operator's name

    Returns:
        A 'Blob' with the same type as 'x' and 'y'
    """
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
    r"""add all input blobs element-wise.

    Args:
        inputs: Sequence of 'Blob'
        name: This operator's name

    Returns:
        A 'Blob' with the same type as 'inputs' Blobs
    """
    if os.getenv("ENABLE_USER_OP") == "False":
        op_conf = op_conf_util.OperatorConf()
        setattr(
            op_conf, "name", name if name is not None else id_util.UniqueStr("AddN_"),
        )
        assert len(inputs) > 1
        for blob in inputs:
            getattr(op_conf.add_conf, "in").append(blob.unique_name)
        op_conf.add_conf.out = "out"
        interpret_util.Forward(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)
    return _recursive_build_add_n(inputs, name)


@oneflow_export("math.subtract")
def subtract(
    x: Union[int, float, remote_blob_util.BlobDef],
    y: Union[int, float, remote_blob_util.BlobDef],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Computes 'x' subtract 'y'

    Args:
        x: A 'Blob' with type of 'int' or 'float' 
        y: A 'Blob' with type of 'int' or 'float' 
        name: This operator's name

    Returns:
        A 'Blob' with the same type as 'x' and 'y'
    """
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
    r"""Computes 'x' multiply 'y'

    Args:
        x: A 'Blob' with type of 'int' or 'float' 
        y: A 'Blob' with type of 'int' or 'float' 
        name: This operator's name

    Returns:
        A 'Blob' with the same type as 'x' and 'y'
    """
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
    r"""Computes 'x' divide 'y'

    Args:
        x: A 'Blob' with type of 'int' or 'float' 
        y: A 'Blob' with type of 'int' or 'float' 
        name: This operator's name

    Returns:
        A 'Blob' with the same type as 'x' and 'y'
    """
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
    r"""Computes 'x' mod 'y'

    Args:
        x: A 'Blob' with type of 'int' or 'float' 
        y: A 'Blob' with type of 'int' or 'float' 
        name: This operator's name

    Returns:
        A 'Blob' with the same type as 'x' and 'y'
    """
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
            builder.Attr("has_int_operand", True, "AttrTypeBool")
            .Attr("has_float_operand", False, "AttrTypeBool")
            .Attr("int_operand", operand, "AttrTypeInt64")
            .Attr("float_operand", 0.0, "AttrTypeDouble")
        )
    elif isinstance(operand, float):
        builder = (
            builder.Attr("has_int_operand", False, "AttrTypeBool")
            .Attr("has_float_operand", True, "AttrTypeBool")
            .Attr("int_operand", 0, "AttrTypeInt64")
            .Attr("float_operand", operand, "AttrTypeDouble")
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
    if os.getenv("ENABLE_USER_OP") != "False":
        return flow.math.add_n([x, y], name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ElementWiseAdd_"),
    )
    getattr(op_conf.add_conf, "in").append(x.unique_name)
    getattr(op_conf.add_conf, "in").append(y.unique_name)
    op_conf.add_conf.out = "out"
    interpret_util.Forward(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


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
            builder.Attr("has_int_operand", True, "AttrTypeBool")
            .Attr("has_float_operand", False, "AttrTypeBool")
            .Attr("int_operand", operand, "AttrTypeInt64")
            .Attr("float_operand", 0.0, "AttrTypeDouble")
        )
    elif isinstance(operand, float):
        builder = (
            builder.Attr("has_int_operand", False, "AttrTypeBool")
            .Attr("has_float_operand", True, "AttrTypeBool")
            .Attr("int_operand", 0, "AttrTypeInt64")
            .Attr("float_operand", operand, "AttrTypeDouble")
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


@oneflow_export("math.tanh", "keras.activations.tanh")
def tanh(
    x: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Computes hyperbolic tangent of `x` element-wise.

    Args:
        x: Input `Blob`.
    Returns:
        A `Blob` with the same type as `x`.
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


@oneflow_export("math.gelu", "keras.activations.gelu")
def gelu(
    x: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Gaussian Error Linear Units.

    Args:
        x: Input `Blob`.
    Returns:
        A `Blob` with the same type as `x`.
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
        A `Blob` with the same type as `x`.
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
        A `Blob` with the same type as `x`.
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
    r"""Computes the sum along segment of 'data'

    Args:
        data: a 'Blob'
        segment_ids: A 'Blob' contains of segment ids
        num_segments: the number of segments
        axis: 
        name: This operator's name

    Returns:
        A `Blob` with the same type as `data`.
    """
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("UnsortedSegmentSum_")
        )
        .Op("unsorted_segment_sum")
        .Input("data", [data])
        .Input("segment_ids", [segment_ids])
        .Output("out")
        .Attr("axis", int(axis), "AttrTypeInt64")
        .Attr("num_segments", int(num_segments), "AttrTypeInt64")
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
    r"""Computes the sum along segment of 'data'

    Args:
        data: a 'Blob'
        segment_ids: A 'Blob' contains of segment ids
        like: The shape for output
        axis:
        name (Optional[str], optional): [description]. Defaults to None.

    Returns:
        A `Blob` with the same shape as `like`.
    """
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("UnsortedSegmentSumLike_")
        )
        .Op("unsorted_segment_sum_like")
        .Input("data", [data])
        .Input("segment_ids", [segment_ids])
        .Input("like", [like])
        .Output("out")
        .Attr("axis", int(axis), "AttrTypeInt64")
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
        .Attr("num_segments", int(num_segments), "AttrTypeInt64")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("cast")
def cast(
    x: remote_blob_util.BlobDef, dtype: int, name: Optional[str] = None
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
        .Attr("dtype", dtype, "AttrTypeDataType")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.equal")
def equal(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Return a 'Blob' of (x == y) element-wise 

    Args:
        x: a 'Blob'
        y: a 'Blob'
        name: This operator's name

    Returns:
        A 'Blob' with value '1' or '0'
    """
    return build_broadcast_binary_op("broadcast_equal", x, y, name)


@oneflow_export("math.not_equal")
def not_equal(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Return a 'Blob' of (x != y) element-wise 

    Args:
        x: a 'Blob'
        y: a 'Blob'
        name: This operator's name

    Returns:
        A 'Blob' with value '1' or '0'
    """
    return build_broadcast_binary_op("broadcast_not_equal", x, y, name)


@oneflow_export("math.less")
def less(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Return a 'Blob' of (x < y) element-wise 

    Args:
        x: a 'Blob'
        y: a 'Blob'
        name: This operator's name

    Returns:
        A 'Blob' with value '1' or '0'
    """
    return build_broadcast_binary_op("broadcast_less", x, y, name)


@oneflow_export("math.less_equal")
def less_equal(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Return a 'Blob' of (x <= y) element-wise 

    Args:
        x: a 'Blob'
        y: a 'Blob'
        name: This operator's name

    Returns:
        A 'Blob' with value '1' or '0'
    """
    return build_broadcast_binary_op("broadcast_less_equal", x, y, name)


@oneflow_export("math.greater")
def greater(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Return a 'Blob' of (x > y) element-wise 

    Args:
        x: a 'Blob'
        y: a 'Blob'
        name: This operator's name

    Returns:
        A 'Blob' with value '1' or '0'
    """
    return build_broadcast_binary_op("broadcast_greater", x, y, name)


@oneflow_export("math.greater_equal")
def greater_equal(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Return a 'Blob' of (x >= y) element-wise 

    Args:
        x: a 'Blob'
        y: a 'Blob'
        name: This operator's name

    Returns:
        A 'Blob' with value '1' or '0'
    """
    return build_broadcast_binary_op("broadcast_greater_equal", x, y, name)


@oneflow_export("math.logical_and")
def logical_and(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Return a 'Blob' of (x && y) element-wise 

    Args:
        x: a 'Blob'
        y: a 'Blob'
        name: This operator's name

    Returns:
        A 'Blob' with value '1' or '0'
    """
    return build_broadcast_binary_op("broadcast_logical_and", x, y, name)


@oneflow_export("math.minimum")
def broadcast_min(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Return a 'Blob' with the min value of (x < y ? x : y) element-wise 

    Args:
        x: a 'Blob'
        y: a 'Blob'
        name: This operator's name

    Returns:
        A 'Blob' with min value
    """
    return build_broadcast_binary_op("broadcast_minimum", x, y, name)


@oneflow_export("math.maximum")
def broadcast_max(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Return a 'Blob' with the max value of (x > y ? x : y) element-wise 

    Args:
        x: a 'Blob'
        y: a 'Blob'
        name: This operator's name

    Returns:
        A 'Blob' with max value
    """
    return build_broadcast_binary_op("broadcast_maximum", x, y, name)


@oneflow_export("math.reduced_shape_elem_cnt")
def elem_cnt(
    input_blob: remote_blob_util.BlobDef,
    axis: Optional[Sequence[int]] = None,
    dtype: Optional[int] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Reduce shape emements count accross 'axis'

    Args:
        input_blob: a 'Blob'
        axis: The axis need to be reduced
        dtype: 'int'
        name: This operator's name

    Returns:
        A 'Blob'
    """
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
        op_conf.shape_elem_cnt_conf.data_type = dtype
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
    r"""Returns a 'Blob' whose value is the index of top-k values accross the last axes of 'input'

    Args:
        input: a 'Blob'
        k: Number of top elements to look for along the last dimension
        sorted: 'True' or 'False'
        name: This operator's name

    Returns:
        A 'Blob' with the index 'input'
    """
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("TopK_"))
        .Op("top_k")
        .Input("in", [input])
        .Output("out")
        .Attr("k", k, "AttrTypeInt32",)
        .Attr("sorted", sorted, "AttrTypeBool",)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.argmax")
def argmax(
    input: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Returns a 'Blob' whose value is the index with largest value accross the last axes of 'input'

    Args:
        input: a 'Blob'
        name: This operator's name

    Returns:
        A 'Blob' with type 'flow.int32'
    """
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
    r""" 

    Args:
        x (remote_blob_util.BlobDef): [description]
        compatible (Sequence[remote_blob_util.BlobDef]): [description]
        name (Optional[str], optional): [description]. Defaults to None.

    Returns:
        remote_blob_util.BlobDef: [description]
    """
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
            .Attr("floating_min", float(min_value), "AttrTypeDouble")
            .Attr("integral_min", int(min_value), "AttrTypeInt64")
            .Attr("floating_max", float(max_value), "AttrTypeDouble")
            .Attr("integral_max", int(max_value), "AttrTypeInt64")
        )
    elif min_value is not None:
        op_builder = (
            flow.user_op_builder(name)
            .Op("clip_by_scalar_min")
            .Attr("floating_min", float(min_value), "AttrTypeDouble")
            .Attr("integral_min", int(min_value), "AttrTypeInt64")
        )
    elif max_value is not None:
        op_builder = (
            flow.user_op_builder(name)
            .Op("clip_by_scalar_max")
            .Attr("floating_max", float(max_value), "AttrTypeDouble")
            .Attr("integral_max", int(max_value), "AttrTypeInt64")
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
    r"""L2 Normalization

    Args:
        input: a 'Blob'
        axis: Dimension along which to normalize
        epsilon: a lower bound value for the norm
        name: This operator's name

    Returns:
        A 'Blob' with the same shape as 'input'
    """
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
        .Attr("axis", int(axis), "AttrTypeInt32")
        .Attr("epsilon", float(epsilon), "AttrTypeFloat")
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
    r"""Returns (x-y)(x-y) element-wise

    Args:
        x: a 'Blob'
        y: a 'Blob'
        name: This operator's name

    Returns:
        A 'Blob' with type of 'int' or 'float'
    """
    name_subtract, name_square = None, None
    if name is not None:
        name_subtract = name + "_subtract"
        name_square = name + "_square"
    return flow.math.square(flow.math.subtract(x, y, name_subtract), name_square)
