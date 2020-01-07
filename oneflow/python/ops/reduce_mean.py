from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
import collections
import oneflow as flow

@oneflow_export("math.reduce_mean")
def reduce_mean(input_blob, axis=None, keepdims=False, name=None):
    reduce_sum = flow.math.reduce_sum(input_blob, axis=axis, keepdims=keepdims, name=name)
    reduce_count = flow.math.reduced_shape_elem_cnt(input_blob, axis=axis, dtype=input_blob.dtype)
    return reduce_sum / reduce_count
