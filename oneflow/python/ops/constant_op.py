from __future__ import absolute_import

import os

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("constant")
def constant(value, dtype=None, shape=None, name=None):
    is_user_op = False
    if os.getenv("ENABLE_USER_OP") == "True":
        is_user_op = True
    if name is None:
        name = id_util.UniqueStr("Constant_")
    assert value is not None
    assert dtype is not None

    if not isinstance(value, (int, float)):
        raise NotImplementedError

    if is_user_op:
        if isinstance(value, float):
            is_floating_value = True
        else:
            is_floating_value = False
        if shape is not None:
            assert isinstance(shape, (list, tuple))
        else:
            shape = []
        return (
            flow.user_op_builder(name)
            .Op("constant")
            .Output("out")
            .Attr("floating_value", float(value), "AttrTypeDouble")
            .Attr("integer_value", int(value), "AttrTypeInt64")
            .Attr("is_floating_value", is_floating_value, "AttrTypeBool")
            .Attr("dtype", dtype, "AttrTypeDataType")
            .Attr("shape", shape, "AttrTypeShape")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
        op_conf = op_conf_util.OperatorConf()
        setattr(op_conf, "name", name)
        setattr(op_conf.constant_conf, "data_type", dtype)
        op_conf.constant_conf.initializer.CopyFrom(
            flow.constant_initializer(value, dtype)
        )
        if shape is not None:
            assert isinstance(shape, (list, tuple))
            op_conf.constant_conf.shape.dim.extend(list(shape))

        setattr(op_conf.constant_conf, "out", "out")
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("constant_scalar")
def constant_scalar(value, dtype=None, name=None):
    return flow.constant(value, dtype=dtype, shape=[1])


@oneflow_export("constant_like")
def constant_like(like, value, dtype=None, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ConstantLike_"),
    )
    setattr(op_conf.constant_like_conf, "like", like.unique_name)
    if isinstance(value, int):
        op_conf.constant_like_conf.int_operand = value
    elif isinstance(value, float):
        op_conf.constant_like_conf.float_operand = value
    else:
        raise NotImplementedError
    if dtype is not None:
        setattr(op_conf.constant_like_conf, "data_type", dtype)
    setattr(op_conf.constant_like_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("ones_like")
def ones_like(like, dtype=None, name=None):
    return constant_like(like, 1, dtype=dtype, name=name)


@oneflow_export("zeros_like")
def zeros_like(like, dtype=None, name=None):
    return constant_like(like, 0, dtype=dtype, name=name)
