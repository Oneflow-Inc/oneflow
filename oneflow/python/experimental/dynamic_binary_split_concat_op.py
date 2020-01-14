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
import oneflow.core.job.sbp_parallel_pb2 as sbp_parallel_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export('experimental.dynamic_binary_split')
def dynamic_binary_split(x, base_shift=2, out_num=2, name=None):
    op_conf = op_conf_util.OperatorConf()
    if name is None:
        op_conf.name = id_util.UniqueStr('DynamicBinarySplit_')
    else:
        op_conf.name = name

    obns = []
    out_remote_blobs = []
    for i in range(out_num):
        obns.append("out_" + str(i))

    op_conf.dynamic_binary_split_conf.in = x.logical_blob_name
    op_conf.dynamic_binary_split_conf.out[:] = obns

    compile_context.CurJobAddOp(op_conf)
    for i in range(out_num):
        out_lbi = logical_blob_id_util.LogicalBlobId()
        out_lbi.op_name = op_conf.name
        out_lbi.blob_name = obns[i]
        out_remote_blobs.append(remote_blob_util.RemoteBlob(out_lbi))

    return out_remote_blobs

@oneflow_export('experimental.dynamic_binary_concat')
def dynamic_binary_concat(input_blob_list, source_blob, name=None):
    op_conf = op_conf_util.OperatorConf()
    if name is None:
        op_conf.name = id_util.UniqueStr('DynamicBinaryConcat_')
    else:
        op_conf.name = name

    in_lbns = []
    for in_blob in input_blob_list:
        in_lbns.append()

    op_conf.dynamic_binary_split_conf.in = x.logical_blob_name
    op_conf.dynamic_binary_split_conf.out[:] = obns

    compile_context.CurJobAddOp(op_conf)
    for i in range(out_num):
        out_lbi = logical_blob_id_util.LogicalBlobId()
        out_lbi.op_name = op_conf.name
        out_lbi.blob_name = obns[i]
        out_remote_blobs.append(remote_blob_util.RemoteBlob(out_lbi))

    return out_remote_blobs

