from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow
from oneflow.python.oneflow_export import oneflow_export

import collections

@oneflow_export("nn.unsorted_segment_sum")
def unsorted_segment_sum(data, segment_ids, axis=None, num_segments=0, name=None):
    if os.getenv("ENABLE_USER_OP") == 'True':
        if name is None:
            name = id_util.UniqueStr("UnsortedSegmentSum_")
        if axis is None:
            axis = 0
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("gather_grad")\
            .Input("data", [data])\
            .Input("segment_ids", [segment_ids])\
            .Output("out")\
            .SetAttr("axis", int(axis), "AttrTypeInt64")\
            .SetAttr("num_segments", int(num_segments), "AttrTypeInt64")\
            .Build().RemoteBlobList()[0]

