from __future__ import absolute_import

import oneflow.python.framwork.compile_context as compile_context
import oneflow.python.framwork.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framwork.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

def relu(x, alpha=0., max_value=None, threshold=0.)
    assert alpha == 0.
    assert max_value == None
    assert threshold == 0.
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr('Relu_')
    setattr(op_conf.relu_conf, 'in', x.lbn)
    op_conf.relu_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


        
