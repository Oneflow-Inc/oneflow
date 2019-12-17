from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.input_blob_def as input_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

def InputOpByArgBlobDef(blob_def):
    assert isinstance(blob_def, input_blob_util.ArgBlobDef)
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = blob_def.op_name
    op_conf.input_conf.out = blob_def.blob_name
    op_conf.input_conf.blob_conf.CopyFrom(blob_def.ToInterfaceBlobConf())
    op_conf.input_conf.blob_conf.batch_axis.value = 0
    blob_def.AddAndInferOp(op_conf)
    return remote_blob_util.RemoteBlob(blob_def.lbi)

def RetOpByRemoteBlob(remote_blob):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr('Return_')
    setattr(op_conf.return_conf, 'in', remote_blob.logical_blob_name)
    op_conf.return_conf.out = "out"
    compile_context.CurJobAddOp(op_conf, remote_blob.parallel_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
