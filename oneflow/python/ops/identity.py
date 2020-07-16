from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
from typing import Optional


@oneflow_export("sleep_identity")
def sleep_identity(
    x: remote_blob_util.BlobDef, seconds: int, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    if name is None:
        name = id_util.UniqueStr("Sleep_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.sleep_conf, "in", x.unique_name)
    op_conf.sleep_conf.seconds = seconds
    op_conf.sleep_conf.out = "out"
    interpret_util.Forward(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
