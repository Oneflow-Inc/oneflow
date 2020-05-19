from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.ops.user_op_builder as user_op_builder
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.distribute as distribute_util

import oneflow as flow

@oneflow_export("math.two_stage_reduce_max")
def two_stage_reduce_max(x, axis=None, keepdims=False, name=None):
    name = name if name is not None else id_util.UniqueStr("ReduceMax_")
    two_stage_reduce(x, axis, keepdims, "reduce_max",name)

@oneflow_export("math.two_stage_reduce_min")
def two_stage_reduce_min(x, axis=None, keepdims=False, name=None):
    name = name if name is not None else id_util.UniqueStr("ReduceMin_")
    two_stage_reduce(x, axis, keepdims, "reduce_min", name)

def two_stage_reduce(x, axis=None, keepdims=False, op_type_name=None, name=None):

    assert check_x_dictribute(x, axis)

    device_stage_out_list=[]
    device_stage_count_list=[]
    x_list = flow.advanced.distribute_split(x)
    for i in range(4):
      with flow.device_prior_placement("gpu", "0:" + str(i)):
          device_stage_out, device_stage_count = reduce_device_stage(x_list[i], axis, op_type_name+"_device_stage", name+"_device_stage"+str(i))
          device_stage_out_list.append(device_stage_out)
          device_stage_count_list.append(device_stage_count)

    device_stage_out = flow.advanced.distribute_concat(device_stage_out_list)
    device_stage_count = flow.advanced.distribute_concat(device_stage_count_list)

    device_stage_out = device_stage_out.with_distribute(flow.distribute.broadcast())
    device_stage_count = device_stage_count.with_distribute(flow.distribute.broadcast())
    
    out = reduce_global_stage(device_stage_out, device_stage_count, axis, keepdims, op_type_name+"_global_stage", name+"_global_stage")
    return out


def reduce_device_stage(x, axis, op_name, name):
    out, mask, max_count = (flow.user_op_builder(name).Op(op_name)
        .Input("in", [x])
        .Output("out")
        .Output("mask")
        .Output("max_count")
        .Attr("axis", axis, "AttrTypeListInt32")
        .Build().InferAndTryRun().RemoteBlobList())
    return out, max_count

def reduce_global_stage(x, device_max_count, axis, keepdims, op_name, name):
    out, mask = (flow.user_op_builder(name).Op(op_name)
        .Input("in", [x])
        .Input("device_max_count", [device_max_count])
        .Output("out")
        .Output("mask")
        .Attr("axis", axis, "AttrTypeListInt32")
        .Attr("keepdims", keepdims, "AttrTypeBool")
        .Build().InferAndTryRun().RemoteBlobList())
    return out

def check_x_dictribute(x, axis):
    for i in axis:
        if x.distribute is distribute_util.split(i):
            return True
    return False
