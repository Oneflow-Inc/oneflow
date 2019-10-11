from __future__ import absolute_import
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("debug.distribute_split")
def distribute_split(x, axis=0, name=None):
    import oneflow
    if name is None: name = id_util.UniqueStr("DistributeConcat_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.distribute_split_conf, "in", x.logical_blob_name)
    op_conf.distribute_split_conf.axis = axis
    ret = []
    for i in range(oneflow.current_placement_scope.parallel_size()):
        out = "out_%d" % i
        op_conf.distribute_split_conf.out.append(out)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = out
        ret.append(remote_blob_util.RemoteBlob(lbi))
    compile_context.CurJobAddOp(op_conf)
    return tuple(ret)
