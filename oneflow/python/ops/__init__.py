from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.input_blob_def as input_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.g_func_ctx as g_func_ctx
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.job.placement_pb2 as placement_proto_pb
import re

def InputOpByArgBlobDef(blob_def):
    assert isinstance(blob_def, input_blob_util.ArgBlobDef)
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = blob_def.op_name
    op_conf.input_conf.out = blob_def.blob_name
    op_conf.input_conf.blob_conf.CopyFrom(blob_def.ToInterfaceBlobConf())
    blob_def.AddAndInferOp(op_conf)
    return remote_blob_util.RemoteBlob(blob_def.lbi)

def RetOpByRemoteBlob(remote_blob, allow_cpu_return_op = True):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr('Return_')
    setattr(op_conf.return_conf, 'in', remote_blob.logical_blob_name)
    op_conf.return_conf.out = "out"
    parallel_conf = placement_proto_pb.ParallelConf()
    parallel_conf.CopyFrom(remote_blob.parallel_conf)
    if allow_cpu_return_op:
        op_conf.device_type = g_func_ctx.DeviceType4DeviceTag('cpu')
        for i in range(len(parallel_conf.device_name)):
            parallel_conf.device_name[i] = re.sub(":\w+:", ":cpu:", parallel_conf.device_name[i])
    compile_context.CurJobAddOp(op_conf, parallel_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
