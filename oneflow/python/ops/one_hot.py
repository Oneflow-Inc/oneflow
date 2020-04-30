from __future__ import absolute_import

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
from oneflow.python.oneflow_export import oneflow_export
import os


@oneflow_export("one_hot")
def one_hot(
    indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None
):  
    if on_value is None:
        on_value = 1
    if off_value is None:
        off_value = 0

    if os.getenv("ENABLE_USER_OP") == 'True':
        return (
            flow.user_op_builder(name if name is not None else id_util.UniqueStr("OneHot_"))
            .Op("one_hot")
            .Input("indices", [indices])
            .SetAttr("depth", int(depth), "AttrTypeInt64")
            .SetAttr("on_value", float(on_value), "AttrTypeFloat")
            .SetAttr("off_value", float(off_value), "AttrTypeFloat")
            .SetAttr("dtype", dtype, "AttrTypeDataType")
            .Output("out")
            .Build()
            .RemoteBlobList()[0]
        )