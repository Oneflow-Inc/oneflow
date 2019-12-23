from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("math.add")
def add(x, y, name=None):
    if isinstance(x, (int, float)):
        return scalar_add(y, x, name)
    elif isinstance(y, (int, float)):
        return scalar_add(x, y, name)
    elif x.static_shape == y.static_shape and x.batch_axis == y.batch_axis:
        return element_wise_add(x, y, name)
    else:
        return broadcast_add(x, y, name)


@oneflow_export("math.subtract")
def subtract(x, y, name=None):
    if isinstance(x, (int, float)):
        return scalar_add(-1 * y, x, name)
    elif isinstance(y, (int, float)):
        return scalar_add(x, -1 * y, name)
    elif x.static_shape == y.static_shape:
        # TODO: add element-wise op
        return broadcast_sub(x, y, name)
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
    else:
        return broadcast_div(x, y, name)


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
        op_conf, "name", name if name is not None else id_util.UniqueStr("BroadcastAdd_")
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
        op_conf, "name", name if name is not None else id_util.UniqueStr("BroadcastSub_")
    )
    op_conf.broadcast_sub_conf.a = x.logical_blob_name
    op_conf.broadcast_sub_conf.b = y.logical_blob_name
    op_conf.broadcast_sub_conf.out = "out"
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
        op_conf, "name", name if name is not None else id_util.UniqueStr("BroadcastMul_")
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


def broadcast_div(x, y, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("BroadcastDiv_")
    )
    op_conf.broadcast_div_conf.a = x.logical_blob_name
    op_conf.broadcast_div_conf.b = y.logical_blob_name
    op_conf.broadcast_div_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export('math.tanh')
def tanh(x, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr('TanH_'))
    setattr(op_conf.tanh_conf, "in", x.logical_blob_name)
    setattr(op_conf.tanh_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export('math.gelu')
def gelu(x, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr('Gelu_'))
    setattr(op_conf.gelu_conf, "in", x.logical_blob_name)
    setattr(op_conf.gelu_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export('math.relu')
def relu(x, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr('Relu_'))
    setattr(op_conf.relu_conf, "in", x.logical_blob_name)
    setattr(op_conf.relu_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export('math.sigmoid')
def sigmoid(x, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr('Sigmoid_'))
    setattr(op_conf.sigmoid_conf, "in", x.logical_blob_name)
    setattr(op_conf.sigmoid_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("math.unsorted_segment_sum", "unsorted_segment_sum")
def unsorted_segment_sum(data, segment_ids, num_segments, name=None):
    if name is None: name = id_util.UniqueStr("UnsortedSegmentSum_")
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
    if name is None: name = id_util.UniqueStr("UnsortedBatchSegmentSum_")

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

def sqrt(x, name=None):
    # TODO: not ready yet
    raise NotImplementedError
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr('Sqrt_'))
    setattr(op_conf.sqrt_conf, "in", x.logical_blob_name)
    setattr(op_conf.sqrt_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


def rsqrt(x, name=None):
    # TODO: not ready yet
    raise NotImplementedError
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr('Rsqrt_'))
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
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr('Cast_'))
    setattr(op_conf.cast_conf, "in", x.logical_blob_name)
    setattr(op_conf.cast_conf, "data_type", dtype)
    setattr(op_conf.cast_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export('math.leaky_relu')
def leaky_relu(x, alpha=None, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr('LeakyRelu_'))
    setattr(op_conf.leaky_relu_conf, "in", x.logical_blob_name)
    setattr(op_conf.leaky_relu_conf, "out", "out")
    if alpha is not None:
        setattr(op_conf.leaky_relu_conf, "alpha", alpha)
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("math.top_k")
def top_k(input, k=1, sorted=True, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("TopK_"))
    setattr(op_conf.top_k_conf, "in", input.logical_blob_name)
    setattr(op_conf.top_k_conf, "k", k)
    setattr(op_conf.top_k_conf, "sorted", sorted)
    setattr(op_conf.top_k_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)
