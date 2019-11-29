from __future__ import absolute_import
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export
import oneflow

@oneflow_export("advanced.distribute_clone", "debug.distribute_clone")
def distribute_clone(x, name=None):
    if name is None: name = id_util.UniqueStr("DistributeClone_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.distribute_clone_conf, "in", x.logical_blob_name)
    parallel_size = oneflow.placement.current_scope().parallel_size
    op_conf.distribute_clone_conf.out.extend(["out_%d" % i for i in range(parallel_size)])
    compile_context.CurJobAddOp(op_conf)
    ret = []
    for i in range(parallel_size):
        out = "out_%d" % i
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = out
        ret.append(remote_blob_util.RemoteBlob(lbi))
    return tuple(ret)
