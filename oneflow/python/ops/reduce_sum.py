from __future__ import absolute_import

import os
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow
import collections


@oneflow_export("math.reduce_sum")
def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
    if name is None:
        name = id_util.UniqueStr("ReduceSum_")
    enable_user_op = os.getenv("ENABLE_USER_OP") == "True"
    if enable_user_op and flow.current_global_function_desc().IsTrainable() == False:
        if axis is None:
            axis = []
        elif isinstance(axis, (list, tuple)):
           if len(axis) == 0: return input_tensor
        else:
           assert type(axis) is int 
           axis = [axis]
        return (
            flow.user_op_builder(name)
            .Op("reduce_sum")
            .Input("input_tensor", [input_tensor])
            .Output("output_tensor")
            .SetAttr("axis", axis, "AttrTypeListInt32")
            .SetAttr("keepdims", keepdims, "AttrTypeBool")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
        op_conf = op_conf_util.OperatorConf()
        setattr(op_conf, "name", name)
        setattr(op_conf.reduce_sum_conf, "in", input_tensor.logical_blob_name)
        setattr(op_conf.reduce_sum_conf, "out", "out")
        if axis is not None:
            if isinstance(axis, list) and len(axis) == 0:
                return input_tensor
            else:
                op_conf.reduce_sum_conf.axis[:] = (
                    list(axis) if isinstance(axis, collections.Sized) else [axis]
                )
        setattr(op_conf.reduce_sum_conf, "keep_dims", keepdims)
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)
