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


@oneflow_export("experimental.dynamic_binary_split")
def dynamic_binary_split(x, base_shift=2, out_num=2, name=None):
    op_conf = op_conf_util.OperatorConf()
    if name is None:
        op_conf.name = id_util.UniqueStr("DynamicBinarySplit_")
    else:
        op_conf.name = name

    obns = []
    out_remote_blobs = []
    for i in range(out_num):
        obns.append("out_" + str(i))

    setattr(op_conf.dynamic_binary_split_conf, "in", x.unique_name)
    # op_conf.dynamic_binary_split_conf.in = x.unique_name
    op_conf.dynamic_binary_split_conf.out[:] = obns
    op_conf.dynamic_binary_split_conf.base_shift = base_shift

    compile_context.CurJobAddOp(op_conf)
    for i in range(out_num):
        out_lbi = logical_blob_id_util.LogicalBlobId()
        out_lbi.op_name = op_conf.name
        out_lbi.blob_name = obns[i]
        out_remote_blobs.append(remote_blob_util.RemoteBlob(out_lbi))

    return out_remote_blobs


@oneflow_export("experimental.dynamic_binary_concat")
def dynamic_binary_concat(input_blob_list, source_blob, source_sbp="S:0", name=None):
    op_conf = op_conf_util.OperatorConf()
    if name is None:
        op_conf.name = id_util.UniqueStr("DynamicBinaryConcat_")
    else:
        op_conf.name = name

    in_lbns = []
    for in_blob in input_blob_list:
        in_lbns.append(in_blob.unique_name)

    getattr(op_conf.dynamic_binary_concat_conf, "in").extend(in_lbns)
    # op_conf.dynamic_binary_concat_conf.in[:] = in_lbns
    op_conf.dynamic_binary_concat_conf.out = "out"
    op_conf.dynamic_binary_concat_conf.out_data_type = source_blob.dtype
    op_conf.dynamic_binary_concat_conf.out_shape.dim.extend(list(source_blob.shape))
    if source_blob.batch_axis is not None:
        op_conf.dynamic_binary_concat_conf.out_batch_axis.value = source_blob.batch_axis
    else:
        op_conf.dynamic_binary_concat_conf.out_batch_axis.SetInParent()
    if "S" in source_sbp:
        axis = int(source_sbp.split(":")[-1])
        op_conf.dynamic_binary_concat_conf.out_sbp.split_parallel.axis = axis
    elif "B" in source_sbp:
        op_conf.dynamic_binary_concat_conf.out_sbp.broadcast_parallel.SetInParent()
    elif "P" in source_sbp:
        op_conf.dynamic_binary_concat_conf.out_sbp.partial_sum_parallel.SetInParent()
    else:
        print("Error! invalid sbp str:", source_sbp)
        op_conf.dynamic_binary_concat_conf.out_sbp.SetInParent()

    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    out_lbi.op_name = op_conf.name
    out_lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(out_lbi)
