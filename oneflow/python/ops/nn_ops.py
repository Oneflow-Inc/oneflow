from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
from oneflow.python.oneflow_export import oneflow_export

import collections


@oneflow_export("nn.max_pool1d")
def max_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):
    # TODO: fix cuDNN bugs in pooling_1d
    raise NotImplementedError


@oneflow_export("nn.avg_pool1d")
def avg_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):
    # TODO: fix cuDNN bugs in pooling_1d
    raise NotImplementedError


@oneflow_export("nn.max_pool2d")
def max_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("MaxPool2D_")
    )
    setattr(op_conf.max_pooling_2d_conf, "in", input.logical_blob_name)
    setattr(op_conf.max_pooling_2d_conf, "out", "out")
    op_conf.max_pooling_2d_conf.pool_size[:] = _GetSequence(ksize, 2, "ksize")
    op_conf.max_pooling_2d_conf.strides[:] = _GetSequence(ksize, 2, "strides")
    assert padding in ["VALID", "SAME"]
    setattr(op_conf.max_pooling_2d_conf, "padding", padding)
    assert data_format in ["NHWC", "NCHW", "NCHW_VECT_C"]
    setattr(
        op_conf.max_pooling_2d_conf,
        "data_format",
        "channels_last" if data_format is "NHWC" else "channels_first",
    )
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("nn.avg_pool2d")
def avg_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("AveragePool2D_"),
    )
    setattr(op_conf.average_pooling_2d_conf, "in", input.logical_blob_name)
    setattr(op_conf.average_pooling_2d_conf, "out", "out")
    op_conf.average_pooling_2d_conf.pool_size[:] = _GetSequence(ksize, 2, "ksize")
    op_conf.average_pooling_2d_conf.strides[:] = _GetSequence(ksize, 2, "strides")
    assert padding in ["VALID", "SAME"]
    setattr(op_conf.average_pooling_2d_conf, "padding", padding)
    assert data_format in ["NHWC", "NCHW", "NCHW_VECT_C"]
    setattr(
        op_conf.average_pooling_2d_conf,
        "data_format",
        "channels_last" if data_format is "NHWC" else "channels_first",
    )
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("nn.max_pool3d")
def max_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("MaxPool3D_")
    )
    setattr(op_conf.max_pooling_3d_conf, "in", input.logical_blob_name)
    setattr(op_conf.max_pooling_3d_conf, "out", "out")
    op_conf.max_pooling_3d_conf.pool_size[:] = _GetSequence(ksize, 3, "ksize")
    op_conf.max_pooling_3d_conf.strides[:] = _GetSequence(strides, 3, "strides")
    assert padding in ["VALID", "SAME"]
    setattr(op_conf.max_pooling_3d_conf, "padding", padding)
    assert data_format in ["NDHWC", "NCDHW"]
    setattr(
        op_conf.max_pooling_3d_conf,
        "data_format",
        "channels_last" if data_format is "NDHWC" else "channels_first",
    )
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("nn.avg_pool3d")
def avg_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("AveragePool3D_"),
    )
    setattr(op_conf.average_pooling_3d_conf, "in", input.logical_blob_name)
    setattr(op_conf.average_pooling_3d_conf, "out", "out")
    op_conf.average_pooling_3d_conf.pool_size[:] = _GetSequence(ksize, 3, "ksize")
    op_conf.average_pooling_3d_conf.strides[:] = _GetSequence(strides, 3, "strides")
    assert padding in ["VALID", "SAME"]
    setattr(op_conf.average_pooling_3d_conf, "padding", padding)
    assert data_format in ["NDHWC", "NCDHW"]
    setattr(
        op_conf.average_pooling_3d_conf,
        "data_format",
        "channels_last" if data_format is "NDHWC" else "channels_first",
    )
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


def _GetSequence(value, n, name):
    """Formats value from input"""
    if value is None:
        value = [1]
    elif not isinstance(value, collections.Sized):
        value = [value]

    current_n = len(value)
    if current_n == 1:
        return list(value * n)
    elif current_n == n:
        return list(value)
    else:
        raise ValueError(
            "{} should be of length 1 or {} but was {}".format(name, n, current_n)
        )
