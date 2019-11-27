from __future__ import absolute_import
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export
import oneflow

@oneflow_export("advanced.distribute_concat", "debug.distribute_concat")
def distribute_concat(xs, axis=0, name=None):
    assert oneflow.placement.current_scope().parallel_size == len(xs)
    if name is None: name = id_util.UniqueStr("DistributeConcat_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    getattr(op_conf.distribute_concat_conf, "in").extend([x.logical_blob_name for x in xs])
    op_conf.distribute_concat_conf.axis = axis
    op_conf.distribute_concat_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
