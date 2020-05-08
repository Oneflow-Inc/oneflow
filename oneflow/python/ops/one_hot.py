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
    indices, depth, on_value=1, off_value=0, axis=-1, dtype=None, name=None
):  
    out_ndims = len(indices.shape) + 1
    if axis < 0: axis += out_ndims
    assert axis >= 0 and axis < out_ndims, ValueError(
            "Expected axis to between [%d, %d).  But received: %d " %(-out_ndims, out_ndims, axis)
            )
    out = (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("OneHot_"))
        .Op("one_hot")
        .Input("indices", [indices])
        .SetAttr("depth", int(depth), "AttrTypeInt64")
        .SetAttr("floating_on_value", float(on_value), "AttrTypeDouble")
        .SetAttr("integer_on_value", int(on_value), "AttrTypeInt64")
        .SetAttr("floating_off_value", float(off_value), "AttrTypeDouble")
        .SetAttr("integer_off_value", int(off_value), "AttrTypeInt64")
        .SetAttr("dtype", dtype, "AttrTypeDataType")
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
    if axis != (out_ndims - 1):
        dim_list = list(range(0, out_ndims))
        dim_list.insert(axis, out_ndims-1)
        dim_list.pop()
        return flow.transpose(out, dim_list)
    else:
        return out
