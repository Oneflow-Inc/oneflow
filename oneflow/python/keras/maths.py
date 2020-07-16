from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
from typing import Optional


@oneflow_export("keras.maths.add")
def add(
    x: remote_blob_util.BlobDef, y: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Returns x + y element-wise.
    Args:
        x: A `Blob`.
        y: A `Blob`. Must have the same type as `x`.
        name: A name for the operation (optional).
    Returns:
        A `Blob`. Has the same type as `x`.
    """

    op_conf = op_conf_util.OperatorConf()
    if name is None:
        op_conf.name = id_util.UniqueStr("Add_")
    else:
        op_conf.name = name
    getattr(op_conf.add_conf, "in").append(x.unique_name)
    getattr(op_conf.add_conf, "in").append(y.unique_name)
    op_conf.add_conf.out = "out"
    interpret_util.Forward(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
