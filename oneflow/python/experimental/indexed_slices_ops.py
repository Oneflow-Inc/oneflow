from __future__ import absolute_import

import operator
from functools import reduce

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("experimental.indexed_slices_reduce_sum")
def indexed_slices_reduce_sum(indices, values, name=None):
    op_conf = op_conf_util.OperatorConf()
    if name is None:
        op_conf.name = id_util.UniqueStr("IndexedSlicesReduceSum_")
    else:
        op_conf.name = name

    op_conf.indexed_slices_reduce_sum_conf.x_indices = indices.unique_name
    op_conf.indexed_slices_reduce_sum_conf.x_values = values.unique_name
    op_conf.indexed_slices_reduce_sum_conf.y_indices = "y_indices"
    op_conf.indexed_slices_reduce_sum_conf.y_values = "y_values"
    op_conf.indexed_slices_reduce_sum_conf.num_unique = "num_unique"

    compile_context.CurJobAddOp(op_conf)
    y_indices_lbi = logical_blob_id_util.LogicalBlobId()
    y_indices_lbi.op_name = op_conf.name
    y_indices_lbi.blob_name = "y_indices"
    y_values_lbi = logical_blob_id_util.LogicalBlobId()
    y_values_lbi.op_name = op_conf.name
    y_values_lbi.blob_name = "y_values"
    num_unique_lbi = logical_blob_id_util.LogicalBlobId()
    num_unique_lbi.op_name = op_conf.name
    num_unique_lbi.blob_name = "num_unique"

    return (
        remote_blob_util.RemoteBlob(y_indices_lbi),
        remote_blob_util.RemoteBlob(y_values_lbi),
        remote_blob_util.RemoteBlob(num_unique_lbi),
    )
