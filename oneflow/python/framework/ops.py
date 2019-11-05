from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export('parallel_cast')
def parallel_cast(
        input,
        name=None, distribute=None, gradient_distribute=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ParallelCast_"),
    )
    op_conf.parallel_cast_conf.out = "out"
    setattr(op_conf.parallel_cast_conf, "in", input.logical_blob_name)

    def to_split_axis(dist):
        split_axis = data_type_util.OptInt64()
        if type(dist) is distribute_util.SplitDistribute:
            split_axis.value = dist.axis
        elif type(dist) is distribute_util.BroadcastDistribute:
            split_axis.ClearField("value")
        else:
            raise NotImplementedError
        return split_axis

    if distribute is not None:
        op_conf.parallel_cast_conf.split_axis.CopyFrom(to_split_axis(distribute))
    if gradient_distribute is not None:
        op_conf.parallel_cast_conf.gradient_split_axis.CopyFrom(to_split_axis(gradient_distribute))
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
