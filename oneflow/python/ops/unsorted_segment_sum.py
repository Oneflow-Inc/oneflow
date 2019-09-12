from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("unsorted_segment_sum")
def unsorted_segment_sum(
    data, segment_ids, num_segments, name=None
):
    if name is None: name = id_util.UniqueStr("UnsortedSegmentSum_")

    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    op_conf.unsorted_segment_sum_conf.data = data.logical_blob_name
    op_conf.unsorted_segment_sum_conf.segment_ids = segment_ids.logical_blob_name
    op_conf.unsorted_segment_sum_conf.num_segments = num_segments
    op_conf.unsorted_segment_sum_conf.out = "out"

    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
