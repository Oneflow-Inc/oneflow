from __future__ import absolute_import

from functools import reduce
import operator

import oneflow as flow
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("nvtx.range_push")
def nvtx_range_push(x, msg, name=None):
    if name is None: name = id_util.UniqueStr("NvtxRangePush_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.nvtx_range_push_conf, "in", x.logical_blob_name)
    setattr(op_conf.nvtx_range_push_conf, "msg", msg)
    op_conf.nvtx_range_push_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("nvtx.range_pop")
def nvtx_range_pop(x, name=None):
    if name is None: name = id_util.UniqueStr("NvtxRangePop_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.nvtx_range_pop_conf, "in", x.logical_blob_name)
    op_conf.nvtx_range_pop_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
