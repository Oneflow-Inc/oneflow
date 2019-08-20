from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('keras.maths.matmul')
def matmul( a,
            b,
            transpose_a=False,
            transpose_b=False,
            adjoint_a=False,
            adjoint_b=False,
            a_is_sparse=False,
            b_is_sparse=False,
            name=None):

    assert adjoint_a == False
    assert adjoint_b == False
    assert a_is_sparse == False
    assert b_is_sparse == False

    op_conf = op_conf_util.OperatorConf()
    if name is None:
        op_conf.name = id_util.UniqueStr('Matmul_')
    else:
        op_conf.name = name
    op_conf.matmul_conf.a = a.logical_blob_name
    op_conf.matmul_conf.b = b.logical_blob_name
    op_conf.matmul_conf.transpose_a = transpose_a
    op_conf.matmul_conf.transpose_b = transpose_b
    op_conf.matmul_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export('keras.maths.add')
def add(x,
        y,
        name=None):

    op_conf = op_conf_util.OperatorConf()
    if name is None:
        op_conf.name = id_util.UniqueStr('Add_')
    else:
        op_conf.name = name
    getattr(op_conf.add_conf, 'in').append(x.logical_blob_name)
    getattr(op_conf.add_conf, 'in').append(y.logical_blob_name)
    op_conf.add_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


