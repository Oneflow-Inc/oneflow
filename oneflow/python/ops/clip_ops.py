from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("clip_by_value")
def clip_by_value(t, clip_value_min=None, clip_value_max=None, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("ClipByValue_")
    )
    setattr(op_conf.clip_by_value_conf, "in", t.logical_blob_name)
    if clip_value_min is not None:
        setattr(op_conf.clip_by_value_conf, "min_val", clip_value_min)
    if clip_value_max is not None:
        setattr(op_conf.clip_by_value_conf, "max_val", clip_value_max)
    setattr(op_conf.clip_by_value_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)
