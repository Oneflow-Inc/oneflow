from __future__ import absolute_import

import os

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("math.add")
def add(x, y, name=None):
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
def add_n(inputs, name=None):
    if os.getenv("ENABLE_USER_OP") != "True":
        op_conf = op_conf_util.OperatorConf()
        setattr(
            op_conf, "name", name if name is not None else id_util.UniqueStr("AddN_"),
        )
        assert len(inputs) > 1
        for blob in inputs:
            getattr(op_conf.add_conf, "in").append(blob.unique_name)
        op_conf.add_conf.out = "out"
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)
    return _recursive_build_add_n(inputs, name)


@oneflow_export("math.subtract")
def subtract(x, y, name=None):
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
def multiply(x, y, name=None):
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
def divide(x, y, name=None):
    if isinstance(x, (int, float)):
        raise NotImplementedError
    elif isinstance(y, (int, float)):
        raise NotImplementedError
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
def floor_mod(x, y, name=None):
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
    if os.getenv("ENABLE_USER_OP") == "True":
        builder = (
            flow.user_op_builder(name).Op("scalar_add").Input("in", [x]).Output("out")
        )
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
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.scalar_add_conf, "in", x.unique_name)
    if isinstance(operand, int):
        op_conf.scalar_add_conf.int_operand = operand
    elif isinstance(operand, float):
        op_conf.scalar_add_conf.float_operand = operand
    op_conf.scalar_add_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def scalar_add_by_tensor(x, scalar, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
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
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ScalarAddByTensor_"),
    )
    setattr(op_conf.scalar_add_by_tensor_conf, "in", x.unique_name)
    setattr(op_conf.scalar_add_by_tensor_conf, "scalar", scalar.unique_name)
    op_conf.scalar_add_by_tensor_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def element_wise_add(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
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
    compile_context.CurJobAddOp(op_conf)
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
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_add", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastAdd_"),
    )
    op_conf.broadcast_add_conf.a = x.unique_name
    op_conf.broadcast_add_conf.b = y.unique_name
    op_conf.broadcast_add_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def broadcast_sub(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_sub", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastSub_"),
    )
    op_conf.broadcast_sub_conf.a = x.unique_name
    op_conf.broadcast_sub_conf.b = y.unique_name
    op_conf.broadcast_sub_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def scalar_sub_by_tensor(x, scalar, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
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
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ScalarSubByTensor_"),
    )
    setattr(op_conf.scalar_sub_by_tensor_conf, "in", x.unique_name)
    setattr(op_conf.scalar_sub_by_tensor_conf, "scalar", scalar.unique_name)
    op_conf.scalar_sub_by_tensor_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def element_wise_mul(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
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
    else:
        op_conf = op_conf_util.OperatorConf()
        setattr(
            op_conf,
            "name",
            name if name is not None else id_util.UniqueStr("ElementWiseMul_"),
        )
        setattr(op_conf.multiply_conf, "in_0", x.unique_name)
        setattr(op_conf.multiply_conf, "in_1", y.unique_name)
        op_conf.multiply_conf.out = "out"
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)


def broadcast_mul(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_mul", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastMul_"),
    )
    op_conf.broadcast_mul_conf.a = x.unique_name
    op_conf.broadcast_mul_conf.b = y.unique_name
    op_conf.broadcast_mul_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def scalar_mul(x, operand, name=None):
    if name is None:
        name = id_util.UniqueStr("ScalarMul_")
    if os.getenv("ENABLE_USER_OP") == "True":
        builder = (
            flow.user_op_builder(name).Op("scalar_mul").Input("in", [x]).Output("out")
        )
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
    else:
        op_conf = op_conf_util.OperatorConf()
        setattr(op_conf, "name", name)
        setattr(op_conf.scalar_mul_conf, "in", x.unique_name)
        if isinstance(operand, int):
            op_conf.scalar_mul_conf.int_operand = operand
        elif isinstance(operand, float):
            op_conf.scalar_mul_conf.float_operand = operand
        op_conf.scalar_mul_conf.out = "out"
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)


def scalar_mul_by_tensor(x, scalar, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
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
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ScalarMulByTensor_"),
    )
    setattr(op_conf.scalar_mul_by_tensor_conf, "in", x.unique_name)
    setattr(op_conf.scalar_mul_by_tensor_conf, "scalar", scalar.unique_name)
    op_conf.scalar_mul_by_tensor_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def broadcast_div(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_div", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastDiv_"),
    )
    op_conf.broadcast_div_conf.a = x.unique_name
    op_conf.broadcast_div_conf.b = y.unique_name
    op_conf.broadcast_div_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def scalar_div_by_tensor(x, scalar, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
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
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ScalarDivByTensor_"),
    )
    setattr(op_conf.scalar_div_by_tensor_conf, "in", x.unique_name)
    setattr(op_conf.scalar_div_by_tensor_conf, "scalar", scalar.unique_name)
    op_conf.scalar_div_by_tensor_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def broadcast_floor_mod(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_floor_mod", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastMod_"),
    )
    op_conf.broadcast_floor_mod_conf.a = x.unique_name
    op_conf.broadcast_floor_mod_conf.b = y.unique_name
    op_conf.broadcast_floor_mod_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.tanh", "keras.activations.tanh")
def tanh(x, name=None):
    r"""Computes hyperbolic tangent of `x` element-wise.

    Args:
        x: Input `Blob`.
    Returns:
        A `Blob`
    """
    if os.getenv("ENABLE_USER_OP") != "True":
        op_conf = op_conf_util.OperatorConf()
        setattr(
            op_conf, "name", name if name is not None else id_util.UniqueStr("TanH_")
        )
        setattr(op_conf.tanh_conf, "in", x.unique_name)
        setattr(op_conf.tanh_conf, "out", "out")
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)

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
def gelu(x, name=None):
    r"""Gaussian Error Linear Units.

    Args:
        x: Input `Blob`.
    Returns:
        A `Blob`
    """
    if os.getenv("ENABLE_USER_OP") == "True":
        return (
            flow.user_op_builder(
                name if name is not None else id_util.UniqueStr("Gelu_")
            )
            .Op("gelu")
            .Input("in", [x])
            .Output("out")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
        op_conf = op_conf_util.OperatorConf()
        setattr(
            op_conf, "name", name if name is not None else id_util.UniqueStr("Gelu_")
        )
        setattr(op_conf.gelu_conf, "in", x.unique_name)
        setattr(op_conf.gelu_conf, "out", "out")
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.relu", "nn.relu")
def relu(x, name=None):
    r"""ReLU activation

    Args:
        x: Input `Blob`.
    Returns:
        A `Blob`
    """
    if os.getenv("ENABLE_USER_OP") != "True":
        op_conf = op_conf_util.OperatorConf()
        setattr(
            op_conf, "name", name if name is not None else id_util.UniqueStr("Relu_")
        )
        setattr(op_conf.relu_conf, "in", x.unique_name)
        setattr(op_conf.relu_conf, "out", "out")
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)

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
def sigmoid(x, name=None):
    r"""Computes sigmoid of `x` element-wise.
    
    Args:
        x: Input `Blob`.
    Returns:
        A `Blob`
    """
    if os.getenv("ENABLE_USER_OP") != "True":
        op_conf = op_conf_util.OperatorConf()
        setattr(
            op_conf, "name", name if name is not None else id_util.UniqueStr("Sigmoid_")
        )
        setattr(op_conf.sigmoid_conf, "in", x.unique_name)
        setattr(op_conf.sigmoid_conf, "out", "out")
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)

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
def unsorted_segment_sum(data, segment_ids, num_segments, axis=0, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
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
    else:
        op_conf = op_conf_util.OperatorConf()
        op_conf.name = (
            name if name is not None else id_util.UniqueStr("UnsortedSegmentSum_")
        )
        op_conf.unsorted_segment_sum_conf.data = data.unique_name
        op_conf.unsorted_segment_sum_conf.segment_ids = segment_ids.unique_name
        op_conf.unsorted_segment_sum_conf.num_segments = num_segments
        op_conf.unsorted_segment_sum_conf.axis = axis
        op_conf.unsorted_segment_sum_conf.out = "out"

        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.unsorted_segment_sum_like", "unsorted_segment_sum_like")
def unsorted_segment_sum_like(data, segment_ids, like, axis=0, name=None):
    if name is None:
        name = id_util.UniqueStr("UnsortedSegmentSumLike_")
    if os.getenv("ENABLE_USER_OP") == "True":
        return (
            flow.user_op_builder(
                name
                if name is not None
                else id_util.UniqueStr("UnsortedSegmentSumLike__")
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
    else:
        op_conf = op_conf_util.OperatorConf()
        op_conf.name = name
        op_conf.unsorted_segment_sum_like_conf.data = data.unique_name
        op_conf.unsorted_segment_sum_like_conf.segment_ids = segment_ids.unique_name
        op_conf.unsorted_segment_sum_like_conf.like = like.unique_name
        op_conf.unsorted_segment_sum_like_conf.axis = axis
        op_conf.unsorted_segment_sum_like_conf.out = "out"

        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.unsorted_batch_segment_sum", "unsorted_batch_segment_sum")
def unsorted_batch_segment_sum(data, segment_ids, num_segments, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return (
            flow.user_op_builder(
                name
                if name is not None
                else id_util.UniqueStr("UnsortedBatchSegmentSum_")
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
    else:
        op_conf = op_conf_util.OperatorConf()
        op_conf.name = (
            name if name is not None else id_util.UniqueStr("UnsortedBatchSegmentSum_")
        )
        op_conf.unsorted_batch_segment_sum_conf.data = data.unique_name
        op_conf.unsorted_batch_segment_sum_conf.segment_ids = segment_ids.unique_name
        op_conf.unsorted_batch_segment_sum_conf.num_segments = num_segments
        op_conf.unsorted_batch_segment_sum_conf.out = "out"

        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("cast")
def cast(x, dtype, name=None):
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
    if os.getenv("ENABLE_USER_OP") == "True":
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
    else:
        op_conf = op_conf_util.OperatorConf()
        setattr(op_conf, "name", name)
        setattr(op_conf.cast_conf, "in", x.unique_name)
        setattr(op_conf.cast_conf, "data_type", dtype)
        setattr(op_conf.cast_conf, "out", "out")
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.naive_logical_and")
def naive_logical_and(lhs, rhs, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("LogicalAnd_")
    )
    setattr(op_conf.logical_and_conf, "lhs", lhs.unique_name)
    setattr(op_conf.logical_and_conf, "rhs", rhs.unique_name)
    setattr(op_conf.logical_and_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("math.equal")
def equal(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_equal", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastEqual_"),
    )
    op_conf.broadcast_equal_conf.a = x.unique_name
    op_conf.broadcast_equal_conf.b = y.unique_name
    op_conf.broadcast_equal_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.not_equal")
def not_equal(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_not_equal", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastNotEqual_"),
    )
    op_conf.broadcast_not_equal_conf.a = x.unique_name
    op_conf.broadcast_not_equal_conf.b = y.unique_name
    op_conf.broadcast_not_equal_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.less")
def less(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_less", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastLessThan_"),
    )
    op_conf.broadcast_less_than_conf.a = x.unique_name
    op_conf.broadcast_less_than_conf.b = y.unique_name
    op_conf.broadcast_less_than_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.less_equal")
def less_equal(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_less_equal", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastLessEqual_"),
    )
    op_conf.broadcast_less_equal_conf.a = x.unique_name
    op_conf.broadcast_less_equal_conf.b = y.unique_name
    op_conf.broadcast_less_equal_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.greater")
def greater(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_greater", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastGreaterThan_"),
    )
    op_conf.broadcast_greater_than_conf.a = x.unique_name
    op_conf.broadcast_greater_than_conf.b = y.unique_name
    op_conf.broadcast_greater_than_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.greater_equal")
def greater_equal(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_greater_equal", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastGreaterEqual_"),
    )
    op_conf.broadcast_greater_equal_conf.a = x.unique_name
    op_conf.broadcast_greater_equal_conf.b = y.unique_name
    op_conf.broadcast_greater_equal_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.logical_and")
def logical_and(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_logical_and", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastLogicalAnd_"),
    )
    op_conf.broadcast_logical_and_conf.a = x.unique_name
    op_conf.broadcast_logical_and_conf.b = y.unique_name
    op_conf.broadcast_logical_and_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.minimum")
def broadcast_min(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_minimum", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastMin_"),
    )
    op_conf.broadcast_minimum_conf.a = x.unique_name
    op_conf.broadcast_minimum_conf.b = y.unique_name
    op_conf.broadcast_minimum_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.maximum")
def broadcast_max(x, y, name=None):
    if os.getenv("ENABLE_USER_OP") == "True":
        return build_broadcast_binary_op("broadcast_maximum", x, y, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastMax_"),
    )
    op_conf.broadcast_maximum_conf.a = x.unique_name
    op_conf.broadcast_maximum_conf.b = y.unique_name
    op_conf.broadcast_maximum_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.reduced_shape_elem_cnt")
def elem_cnt(input_blob, axis=None, dtype=None, name=None):
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
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    out_lbi.op_name = op_conf.name
    out_lbi.blob_name = "y"
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("math.top_k")
def top_k(input, k=1, sorted=True, name=None):
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
def argmax(input, name=None):
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
def broadcast_to_compatible_with(x, compatible, name=None):
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
    compile_context.CurJobAddOp(op_conf)

    ret_lbi = logical_blob_id_util.LogicalBlobId()
    ret_lbi.op_name = op_conf.name
    ret_lbi.blob_name = "y"
    return remote_blob_util.RemoteBlob(ret_lbi)


@oneflow_export(
    "math.clip_by_value", "clip_by_value", "clip_by_scalar", "clip", "clamp"
)
def clip_by_value(values, min_value=None, max_value=None, name=None):
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
def l2_normalize(input, axis=None, epsilon=1e-12, name=None):
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
def squared_difference(x, y, name=None):
    name_subtract, name_square = None, None
    if name is not None:
        name_subtract = name + "_subtract"
        name_square = name + "_square"
    return flow.math.square(flow.math.subtract(x, y, name_subtract), name_square)
