from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.common.data_type_pb2 as data_type_conf_util

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('keras.variables.variable')
def variable(shape,
            tick=None,
            dtype=None,
            initializer=None,
            name=None,
            model_split_axis=None):
    
    assert dtype == None
    assert tick == None
    assert model_split_axis == None
    assert initializer == None

    op_conf = op_conf_util.OperatorConf()
    if name is None:
        op_conf.name = id_util.UniqueStr('Variable_')
    else:
        op_conf.name = name
    for i in shape:
        getattr(op_conf.variable_conf.shape, 'dim').append(i)
    op_conf.variable_conf.data_type = data_type_conf_util.kFloat
    op_conf.variable_conf.initializer.random_uniform_conf.min = 1.0
    op_conf.variable_conf.initializer.random_uniform_conf.max = 2.0
    op_conf.variable_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

