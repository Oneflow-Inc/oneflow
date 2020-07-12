from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("sleep_identity")
def sleep_identity(x, seconds, name=None):
    if name is None:
        name = id_util.UniqueStr("Sleep_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.sleep_conf, "in", x.unique_name)
    op_conf.sleep_conf.seconds = seconds
    op_conf.sleep_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
