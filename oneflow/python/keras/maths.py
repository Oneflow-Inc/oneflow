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
    op_conf.name = id_util.UniqueStr('Relu_')
    op_conf.matmul_conf.a = a.lbn
    op_conf.matmul_conf.b = b.lbn
    op_conf.matmul_conf.transpose_a = transpose_a
    op_conf.matmul_conf.transpose_b = transpose_b
    op_conf.matmul_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


dict_activations = {'sigmoid': op_conf_util.kSigmoid,
                    'relu': op_conf_util.kRelu,
                    'tanh': op_conf_util.kTanH}
@oneflow_export('keras.maths.add')
def add(x,
        y,
        activation=None,
        name=None):

    allowed_activations = { 'sigmoid',
                            'tanh',
                            'relu'}
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr('Add_')
    getattr(op_conf.add_conf, 'in').append(x.lbn)
    getattr(op_conf.add_conf, 'in').append(y.lbn)
    if activation is not None:
        if activation not in allowed_activations:
            raise TypeError('Activation argument not understood!')
        else:
            op_conf.add_conf.activation = dict_activations[activation]
    else:
        op_conf.add_conf.activation = op_conf_util.kNone
    op_conf.add_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)



