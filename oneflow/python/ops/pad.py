from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export

import collections

@oneflow_export('pad')
def pad(x, paddings, constant_value=0, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr('Pad_'))
    setattr(op_conf.pad_conf, "in", x.logical_blob_name)
    setattr(op_conf.pad_conf, "out", "out")
    if isinstance(paddings, (list, tuple)):
        assert len(paddings) == len(x.static_shape), ValueError(
            "paddings must be the same size of input dims"
        )
        zero_padding_num = 0
        for p in paddings:
            assert isinstance(p, (list, tuple)) and len(p) == 2, ValueError(
                "the elem of paddings must be a tuple or a list with length of 2"
            )
            if p[0] or p[1]: 
                break
            else:
                zero_padding_num += 1 
        trim_paddings = []
        for i in range(max(zero_padding_num - 1, 0), len(x.static_shape)):
            assert isinstance(paddings[i], (list, tuple)) and len(paddings[i]) == 2, ValueError(
                "the elem of paddings must be a tuple or a list with length of 2"
            )
            trim_paddings.append(paddings[i][0])
            trim_paddings.append(paddings[i][1])
        op_conf.pad_conf.paddings.extend(trim_paddings)
    else:
        raise ValueError("paddings must be a tuple or a list.")
    op_conf.pad_conf.constant_value = constant_value
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
