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
        return scalar_add(y, x)
    elif isinstance(y, (int, float)):
        return scalar_add(x, y)
    elif x.static_shape == y.static_shape:
        return element_wise_add(x, y)
    else:
        return broadcast_add(x, y)


@oneflow_export("math.subtract")
def subtract(x, y, name=None):
    if isinstance(x, (int, float)):
        return scalar_add(-1 * y, x)
    elif isinstance(y, (int, float)):
        return scalar_add(x, -1 * y)
    elif x.static_shape == y.static_shape:
        # TODO: add element-wise op
        return broadcast_sub(x, y)
    else:
        return broadcast_sub(x, y)


@oneflow_export("math.multiply")
def multiply(x, y, name=None):
    if isinstance(x, (int, float)):
        return scalar_mul(y, x)
    elif isinstance(y, (int, float)):
        return scalar_mul(x, y)
    elif x.static_shape == y.static_shape:
        # TODO: add element-wise op
        return broadcast_mul(x, y)
    else:
        return broadcast_mul(x, y)


@oneflow_export("math.divide")
def divide(x, y, name=None):
    if isinstance(x, (int, float)):
        raise NotImplementedError
    elif isinstance(y, (int, float)):
        raise NotImplementedError
    elif x.static_shape == y.static_shape:
        # TODO: add element-wise op
        return broadcast_div(x, y)
    else:
        return broadcast_div(x, y)


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
