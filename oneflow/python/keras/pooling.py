from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("keras.pooling.max_pool_2d")
def max_pool_2d(x, pool_size, strides=None, padding="valid", data_format="channels_last"):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", id_util.UniqueStr("MaxPool2D_"))
    setattr(op_conf.max_pooling_2d_conf, "in", x.logical_blob_name)
    setattr(op_conf.max_pooling_2d_conf, "out", "out")
    assert isinstance(pool_size, list)
    op_conf.max_pooling_2d_conf.pool_size[:] = pool_size
    if strides == None:
        op_conf.max_pooling_2d_conf.strides[:] = pool_size
    else:
        assert isinstance(strides, list)
        op_conf.max_pooling_2d_conf.strides[:] = strides
    setattr(op_conf.max_pooling_2d_conf, "padding", padding)
    setattr(op_conf.max_pooling_2d_conf, "data_format", data_format)
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)



# @oneflow_export("keras.layers.MaxPool1D")
# @oneflow_export("keras.layers.AveragePool1D")
# @oneflow_export("keras.layers.MaxPool2D")
# @oneflow_export("keras.layers.AveragePool2D")
# @oneflow_export("keras.layers.MaxPool3D")
# @oneflow_export("keras.layers.AveragePool3D")
