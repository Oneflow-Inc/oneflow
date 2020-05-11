from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.ops.user_op_builder as user_op_builder
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.distribute as distribute_util

import oneflow as flow


#@oneflow_export("math.two_stage_reduce_max")
#def two_stage_reduce_max(x, axis=None, keepdims=False, name=None):
#    name = name if name is not None else id_util.UniqueStr("ReduceMax_")
#
#    assert check_x_dictribute(x, axis)
#    
#    device_stage_out, device_stage_max_count = reduce_device_stage(x, axis, "reduce_max_device_stage", name+"_device_stage")
#    
#    device_stage_out = device_stage_out.with_distribute(flow.distribute.broadcast())
#    device_stage_max_count = device_stage_max_count.with_distribute(flow.distribute.broadcast())
#    
#    out = reduce_global_stage(device_stage_out, device_stage_max_count, axis, keepdims, "reduce_max_global_stage", name+"_global_stage")
#    return out

@oneflow_export("math.two_stage_reduce_min")
def two_stage_reduce_min(x, axis=None, keepdims=False, name=None):
    name = name if name is not None else id_util.UniqueStr("ReduceMax_")

    assert check_x_dictribute(x, axis)
    
    device_stage_out, device_stage_max_count = reduce_device_stage(x, axis, "reduce_min_device_stage", name+"_device_stage")
    
    device_stage_out = device_stage_out.with_distribute(flow.distribute.broadcast())
    device_stage_max_count = device_stage_max_count.with_distribute(flow.distribute.broadcast())
    
    out = reduce_global_stage(device_stage_out, device_stage_max_count, axis, keepdims, "reduce_min_global_stage", name+"_global_stage")
    return out


@oneflow_export("math.two_stage_reduce_max")
def two_stage_reduce_max(x, axis=None, keepdims=False, name=None):
    name = name if name is not None else id_util.UniqueStr("ReduceMax_")

    assert check_x_dictribute(x, axis)

    device_stage_out_list=[]
    device_stage_max_count_list=[]
    x_list = flow.advanced.distribute_split(x)
    for i in range(4):
      with flow.device_prior_placement("gpu", "0:" + str(i)):
          device_stage_out, device_stage_max_count = reduce_device_stage(x_list[i], axis, "reduce_max_device_stage", name+"_device_stage"+str(i))
          device_stage_out_list.append(device_stage_out)
          device_stage_max_count_list.append(device_stage_max_count)

    device_stage_out = flow.advanced.distribute_concat(device_stage_out_list)
    device_stage_max_count = flow.advanced.distribute_concat(device_stage_max_count_list)

    device_stage_out = device_stage_out.with_distribute(flow.distribute.broadcast())
    device_stage_max_count = device_stage_max_count.with_distribute(flow.distribute.broadcast())
    
    print("device_stage_out.dtype", device_stage_out.dtype)
    print("device_stage_max_count.dtype", device_stage_max_count.dtype)
    out = reduce_global_stage(device_stage_out, device_stage_max_count, axis, keepdims, "reduce_max_global_stage", name+"_global_stage")
    return out





def reduce_device_stage(x, axis, op_name, name):
    out, mask, max_count = (flow.user_op_builder(name).Op(op_name)
        .Input("in", [x])
        .Output("out")
        .Output("mask")
        .Output("max_count")
        .SetAttr("axis", axis, "AttrTypeListInt32")
        .Build().InferAndTryRun().RemoteBlobList())
    return out, max_count

def reduce_global_stage(x, device_max_count, axis, keepdims, op_name, name):
    out, mask = (flow.user_op_builder(name).Op(op_name)
        .Input("in", [x])
        .Input("device_max_count", [device_max_count])
        .Output("out")
        .Output("mask")
        .SetAttr("axis", axis, "AttrTypeListInt32")
        .SetAttr("keepdims", keepdims, "AttrTypeBool")
        .Build().InferAndTryRun().RemoteBlobList())
    return out

def check_x_dictribute(x, axis):
    for i in axis:
        if x.distribute is distribute_util.split(i):
            return True
    return False
