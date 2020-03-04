from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow

@oneflow_export("math.add")
def add(x, y, name=None):
    if isinstance(x, (int, float)):
        return scalar_add(y, x, name)
    elif isinstance(y, (int, float)):
        return scalar_add(x, y, name)
    elif x.static_shape == y.static_shape and x.batch_axis == y.batch_axis:
        return element_wise_add(x, y, name)
    elif x.static_shape == (1,):
        return scalar_add_by_tensor(y, x, name)
    elif y.static_shape == (1,):
        return scalar_add_by_tensor(x, y, name)
    else:
        return broadcast_add(x, y, name)

@oneflow_export("math.add_n")
def add_n(inputs, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("AddN_"),
    )
    assert len(inputs) > 1
    for blob in inputs:
        getattr(op_conf.add_conf, "in").append(blob.logical_blob_name)
    op_conf.add_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("math.subtract")
def subtract(x, y, name=None):
    if isinstance(x, (int, float)):
        return scalar_add(-1 * y, x, name)
    elif isinstance(y, (int, float)):
        return scalar_add(x, -1 * y, name)
    elif x.static_shape == y.static_shape:
        # TODO: add element-wise op
        return broadcast_sub(x, y, name)
    elif x.static_shape == (1, ):
        return scalar_sub_by_tensor(y, x, name)
    elif y.static_shape == (1, ):
        return scalar_sub_by_tensor(x, y, name)
    else:
        return broadcast_sub(x, y, name)


@oneflow_export("math.multiply")
def multiply(x, y, name=None):
    if isinstance(x, (int, float)):
        return scalar_mul(y, x, name)
    elif isinstance(y, (int, float)):
        return scalar_mul(x, y, name)
    elif x.static_shape == y.static_shape and x.batch_axis == y.batch_axis:
        return element_wise_mul(x, y, name)
    elif x.static_shape == (1, ):
        return scalar_mul_by_tensor(y, x, name)
    elif y.static_shape == (1, ):
        return scalar_mul_by_tensor(x, y, name)
    else:
        return broadcast_mul(x, y, name)


@oneflow_export("math.divide")
def divide(x, y, name=None):
    if isinstance(x, (int, float)):
        raise NotImplementedError
    elif isinstance(y, (int, float)):
        raise NotImplementedError
    elif x.static_shape == y.static_shape:
        # TODO: add element-wise op
        return broadcast_div(x, y, name)
    elif x.static_shape == (1, ):
        return scalar_div_by_tensor(y, x, name)
    elif y.static_shape == (1, ):
        return scalar_div_by_tensor(x, y, name)
    else:
        return broadcast_div(x, y, name)


@oneflow_export("math.mod")
def floor_mod(x, y, name=None):
    if isinstance(x, (int, float)):
        raise NotImplementedError
    elif isinstance(y, (int, float)):
        raise NotImplementedError
    elif x.static_shape == y.static_shape:
        # TODO: add element-wise op
        return broadcast_floor_mod(x, y, name)
    else:
        return broadcast_floor_mod(x, y, name)


def scalar_add(x, operand, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("ScalarAdd_")
    )
    setattr(op_conf.scalar_add_conf, "in", x.logical_blob_name)
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
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("ScalarAddByTensor_")
    )
    setattr(op_conf.scalar_add_by_tensor_conf, "in", x.logical_blob_name)
    setattr(op_conf.scalar_add_by_tensor_conf, "scalar", scalar.logical_blob_name)
    op_conf.scalar_add_by_tensor_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

def element_wise_add(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ElementWiseAdd_"),
    )
    getattr(op_conf.add_conf, "in").append(x.logical_blob_name)
    getattr(op_conf.add_conf, "in").append(y.logical_blob_name)
    op_conf.add_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def broadcast_add(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastAdd_"),
    )
    op_conf.broadcast_add_conf.a = x.logical_blob_name
    op_conf.broadcast_add_conf.b = y.logical_blob_name
    op_conf.broadcast_add_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def broadcast_sub(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastSub_"),
    )
    op_conf.broadcast_sub_conf.a = x.logical_blob_name
    op_conf.broadcast_sub_conf.b = y.logical_blob_name
    op_conf.broadcast_sub_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

def scalar_sub_by_tensor(x, scalar, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("ScalarSubByTensor_")
    )
    setattr(op_conf.scalar_sub_by_tensor_conf, "in", x.logical_blob_name)
    setattr(op_conf.scalar_sub_by_tensor_conf, "scalar", scalar.logical_blob_name)
    op_conf.scalar_sub_by_tensor_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

def element_wise_mul(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ElementWiseMul_"),
    )
    setattr(op_conf.multiply_conf, "in_0", x.logical_blob_name)
    setattr(op_conf.multiply_conf, "in_1", y.logical_blob_name)
    op_conf.multiply_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def broadcast_mul(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastMul_"),
    )
    op_conf.broadcast_mul_conf.a = x.logical_blob_name
    op_conf.broadcast_mul_conf.b = y.logical_blob_name
    op_conf.broadcast_mul_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def scalar_mul(x, operand, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("ScalarMul_")
    )
    setattr(op_conf.scalar_mul_conf, "in", x.logical_blob_name)
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
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("ScalarMulByTensor_")
    )
    setattr(op_conf.scalar_mul_by_tensor_conf, "in", x.logical_blob_name)
    setattr(op_conf.scalar_mul_by_tensor_conf, "scalar", scalar.logical_blob_name)
    op_conf.scalar_mul_by_tensor_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

def broadcast_div(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastDiv_"),
    )
    op_conf.broadcast_div_conf.a = x.logical_blob_name
    op_conf.broadcast_div_conf.b = y.logical_blob_name
    op_conf.broadcast_div_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

def scalar_div_by_tensor(x, scalar, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("ScalarDivByTensor_")
    )
    setattr(op_conf.scalar_div_by_tensor_conf, "in", x.logical_blob_name)
    setattr(op_conf.scalar_div_by_tensor_conf, "scalar", scalar.logical_blob_name)
    op_conf.scalar_div_by_tensor_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def broadcast_floor_mod(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastMod_"),
    )
    op_conf.broadcast_floor_mod_conf.a = x.logical_blob_name
    op_conf.broadcast_floor_mod_conf.b = y.logical_blob_name
    op_conf.broadcast_floor_mod_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.tanh")
def tanh(x, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("TanH_"))
    setattr(op_conf.tanh_conf, "in", x.logical_blob_name)
    setattr(op_conf.tanh_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.gelu")
def gelu(x, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Gelu_"))
    setattr(op_conf.gelu_conf, "in", x.logical_blob_name)
    setattr(op_conf.gelu_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.relu")
def relu(x, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Relu_"))
    setattr(op_conf.relu_conf, "in", x.logical_blob_name)
    setattr(op_conf.relu_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.sigmoid")
def sigmoid(x, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("Sigmoid_")
    )
    setattr(op_conf.sigmoid_conf, "in", x.logical_blob_name)
    setattr(op_conf.sigmoid_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.unsorted_segment_sum", "unsorted_segment_sum")
def unsorted_segment_sum(data, segment_ids, num_segments, name=None):
    if name is None:
        name = id_util.UniqueStr("UnsortedSegmentSum_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    op_conf.unsorted_segment_sum_conf.data = data.logical_blob_name
    op_conf.unsorted_segment_sum_conf.segment_ids = segment_ids.logical_blob_name
    op_conf.unsorted_segment_sum_conf.num_segments = num_segments
    op_conf.unsorted_segment_sum_conf.axis = 0
    op_conf.unsorted_segment_sum_conf.out = "out"

    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.unsorted_batch_segment_sum", "unsorted_batch_segment_sum")
def unsorted_batch_segment_sum(data, segment_ids, num_segments, name=None):
    if name is None:
        name = id_util.UniqueStr("UnsortedBatchSegmentSum_")

    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    op_conf.unsorted_batch_segment_sum_conf.data = data.logical_blob_name
    op_conf.unsorted_batch_segment_sum_conf.segment_ids = segment_ids.logical_blob_name
    op_conf.unsorted_batch_segment_sum_conf.num_segments = num_segments
    op_conf.unsorted_batch_segment_sum_conf.out = "out"

    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.sqrt")
def sqrt(x, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Sqrt_"))
    setattr(op_conf.sqrt_conf, "in", x.logical_blob_name)
    setattr(op_conf.sqrt_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.rsqrt")
def rsqrt(x, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Rsqrt_"))
    setattr(op_conf.rsqrt_conf, "in", x.logical_blob_name)
    setattr(op_conf.rsqrt_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("cast")
def cast(x, dtype, name=None):
    if x.dtype == dtype:
        return x
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Cast_"))
    setattr(op_conf.cast_conf, "in", x.logical_blob_name)
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
    setattr(op_conf.logical_and_conf, "lhs", lhs.logical_blob_name)
    setattr(op_conf.logical_and_conf, "rhs", rhs.logical_blob_name)
    setattr(op_conf.logical_and_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("math.equal")
def equal(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastEqual_"),
    )
    op_conf.broadcast_equal_conf.a = x.logical_blob_name
    op_conf.broadcast_equal_conf.b = y.logical_blob_name
    op_conf.broadcast_equal_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.not_equal")
def not_equal(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastNotEqual_"),
    )
    op_conf.broadcast_not_equal_conf.a = x.logical_blob_name
    op_conf.broadcast_not_equal_conf.b = y.logical_blob_name
    op_conf.broadcast_not_equal_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.less")
def less(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastLessThan_"),
    )
    op_conf.broadcast_less_than_conf.a = x.logical_blob_name
    op_conf.broadcast_less_than_conf.b = y.logical_blob_name
    op_conf.broadcast_less_than_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.less_equal")
def less_equal(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastLessEqual_"),
    )
    op_conf.broadcast_less_equal_conf.a = x.logical_blob_name
    op_conf.broadcast_less_equal_conf.b = y.logical_blob_name
    op_conf.broadcast_less_equal_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.greater")
def greater(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastGreaterThan_"),
    )
    op_conf.broadcast_greater_than_conf.a = x.logical_blob_name
    op_conf.broadcast_greater_than_conf.b = y.logical_blob_name
    op_conf.broadcast_greater_than_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.greater_equal")
def greater_equal(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastGreaterEqual_"),
    )
    op_conf.broadcast_greater_equal_conf.a = x.logical_blob_name
    op_conf.broadcast_greater_equal_conf.b = y.logical_blob_name
    op_conf.broadcast_greater_equal_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.logical_and")
def logical_and(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastLogicalAnd_"),
    )
    op_conf.broadcast_logical_and_conf.a = x.logical_blob_name
    op_conf.broadcast_logical_and_conf.b = y.logical_blob_name
    op_conf.broadcast_logical_and_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.minimum")
def broadcast_min(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastMin_"),
    )
    op_conf.broadcast_min_conf.a = x.logical_blob_name
    op_conf.broadcast_min_conf.b = y.logical_blob_name
    op_conf.broadcast_min_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.maximum")
def broadcast_max(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BroadcastMax_"),
    )
    op_conf.broadcast_max_conf.a = x.logical_blob_name
    op_conf.broadcast_max_conf.b = y.logical_blob_name
    op_conf.broadcast_max_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


# only do argmax operation in the last dimension and set output int32_t data type for now
# TODO: support argmax in arbitrary axis and allow user to specify output data type
@oneflow_export("math.argmax")
def argmax(input, axis=None, output_type=None, name=None):
    assert axis is None and output_type is None
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("Argmax_")
    )
    setattr(op_conf.argmax_conf, "in", input.logical_blob_name)
    setattr(op_conf.argmax_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)

@oneflow_export("math.reduced_shape_elem_cnt")
def elem_cnt(input_blob, axis=None, dtype=None, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("ShapeElemCnt_"))
    op_conf.shape_elem_cnt_conf.x = input_blob.logical_blob_name
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

@oneflow_export('math.square')
def square(x, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("Square_"),
    )
    setattr(op_conf.square_conf, "in", x.logical_blob_name)
    setattr(op_conf.square_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("math.top_k")
def top_k(input, k=1, sorted=True, name=None):
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("TopK_"))
        .Op("top_k")
        .Input("in", [input])
        .Output("out")
        .SetAttr("k", k, "AttrTypeInt32",)
        .SetAttr("sorted", sorted, "AttrTypeBool",)
        .Build()
        .RemoteBlobList()[0]
    )
