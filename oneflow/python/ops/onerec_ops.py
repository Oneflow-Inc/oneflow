from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("onerec.FieldConf")
class FieldConf(object):
    def __init__(self, key, dtype, static_shape, is_dynamic=False, reshape=None, batch_padding=None):
        assert isinstance(key, str)
        assert isinstance(static_shape, (list, tuple))

        self.key = key
        self.dtype = dtype
        self.static_shape = static_shape
        self.is_dynamic = is_dynamic
        if reshape is not None:
            assert(len(reshape) == len(static_shape))
        self.reshape = reshape
        if batch_padding is not None:
            assert(len(batch_padding) == len(static_shape))
        self.batch_padding = batch_padding

    def to_proto(self):
        field_conf = op_conf_util.DecodeOneRecFieldConf()
        field_conf.key = self.key
        field_conf.data_type = self.dtype
        field_conf.static_shape.dim.extend(self.static_shape)
        field_conf.is_dynamic = self.is_dynamic
        if self.reshape is not None:
            field_conf.reshape.dim.extend(self.reshape)
        if self.batch_padding is not None:
            field_conf.batch_padding.dim.extend(self.batch_padding)
        return field_conf


@oneflow_export("onerec.decode_onerec")
def decode_onerec(files, fields,
                  batch_size=1,
                  buffer_size=16,
                  name=None):
    if name is None:
        name = id_util.UniqueStr("DecodeOneRec_")

    lbis = []

    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name

    op_conf.decode_onerec_conf.file.extend(files)
    op_conf.decode_onerec_conf.batch_size = batch_size
    op_conf.decode_onerec_conf.buffer_size = buffer_size
    for idx, field in enumerate(fields):
        op_conf.decode_onerec_conf.field.extend([field.to_proto()])
        out_blob_name = "out_" + str(idx)
        op_conf.decode_onerec_conf.out.extend([out_blob_name])
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = name
        lbi.blob_name = out_blob_name
        lbis.append(lbi)

    compile_context.CurJobAddOp(op_conf)
    return tuple(map(lambda x: remote_blob_util.RemoteBlob(x), lbis))
