from __future__ import absolute_import

import os

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("matmul", "linalg.matmul")
def matmul(a, b, transpose_a=False, transpose_b=False, name=None):
    r"""
    Analogous to `tf.linalg.matmul <https://www.tensorflow.org/api_docs/python/tf/linalg/matmul>`_

    """
    if os.getenv("ENABLE_USER_OP") == "True":
        assert len(a.shape) == len(b.shape)
        assert len(a.shape) >= 2
        if name is None:
            name = id_util.UniqueStr("Matmul_")
        if len(a.shape) == 2:
            op = (
                flow.user_op_builder(name)
                .Op("matmul")
                .Input("a", [a])
                .Input("b", [b])
                .Output("out")
                .Attr("transpose_a", transpose_a, "AttrTypeBool")
                .Attr("transpose_b", transpose_b, "AttrTypeBool")
                .Build()
            )
        else:
            op = (
                flow.user_op_builder(name)
                .Op("batch_matmul")
                .Input("a", [a])
                .Input("b", [b])
                .Output("out")
                .Attr("transpose_a", transpose_a, "AttrTypeBool")
                .Attr("transpose_b", transpose_b, "AttrTypeBool")
                .Build()
            )
        return op.InferAndTryRun().RemoteBlobList()[0]
    else:
        op_conf = op_conf_util.OperatorConf()
        setattr(
            op_conf, "name", name if name is not None else id_util.UniqueStr("Matmul_")
        )
        setattr(op_conf.matmul_conf, "a", a.unique_name)
        setattr(op_conf.matmul_conf, "b", b.unique_name)
        setattr(op_conf.matmul_conf, "transpose_a", transpose_a)
        setattr(op_conf.matmul_conf, "transpose_b", transpose_b)
        setattr(op_conf.matmul_conf, "out", "out")
        compile_context.CurJobAddOp(op_conf)
        out_lbi = logical_blob_id_util.LogicalBlobId()
        setattr(out_lbi, "op_name", op_conf.name)
        setattr(out_lbi, "blob_name", "out")
        return remote_blob_util.RemoteBlob(out_lbi)
