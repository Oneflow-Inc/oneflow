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

@oneflow_export("nvtx.range_start")
def nvtx_range_start(x, msg, name=None):
    if not isinstance(x, (list, tuple)):
        x = [x]
    assert len(x) > 0, f"not input found for nvtx range: {msg}"
    if name is None: name = id_util.UniqueStr("NvtxRangeStart_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    print("nvtx_range_start", msg, [i.logical_blob_name for i in x])
    getattr(op_conf.nvtx_range_start_conf, "in").extend([i.logical_blob_name for i in x])
    setattr(op_conf.nvtx_range_start_conf, "msg", msg)
    compile_context.CurJobAddOp(op_conf)

@oneflow_export("nvtx.range_end")
def nvtx_range_end(x, msg, name=None):
    if not isinstance(x, (list, tuple)):
        x = [x]
    assert len(x) > 0, f"not input found for nvtx range: {msg}"
    if name is None: name = id_util.UniqueStr("NvtxRangeEnd_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    print("nvtx_range_end", msg, [i.logical_blob_name for i in x])
    getattr(op_conf.nvtx_range_end_conf, "in").extend([i.logical_blob_name for i in x])
    setattr(op_conf.nvtx_range_end_conf, "msg", msg)
    compile_context.CurJobAddOp(op_conf)
