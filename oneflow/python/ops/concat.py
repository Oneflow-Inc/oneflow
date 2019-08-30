from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('concat')
def concat( values,
            axis,
            name=None):
    op_conf = op_conf_util.OperatorConf()
    if not isinstance(values, (list, tuple)):
        values = [values]
    for value in values:
        getattr(op_conf.concat_conf, "in").append(value.logical_blob_name)
    if name is None:
        op_conf.name = id_util.UniqueStr('Concat_')
    else:
        op_conf.name = name
    op_conf.concat_conf.axis = axis
    op_conf.concat_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)




