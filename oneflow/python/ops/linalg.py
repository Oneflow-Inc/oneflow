from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("matmul", "linalg.matmul")
def matmul(a, b, transpose_a=False, transpose_b=False, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Matmul_"))
    setattr(op_conf.matmul_conf, "a", a.logical_blob_name)
    setattr(op_conf.matmul_conf, "b", b.logical_blob_name)
    setattr(op_conf.matmul_conf, "transpose_a", transpose_a)
    setattr(op_conf.matmul_conf, "transpose_b", transpose_b)
    setattr(op_conf.matmul_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)
