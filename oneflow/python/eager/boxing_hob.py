from __future__ import absolute_import

import oneflow.python.lib.core.high_order_bool as high_order_bool


def boxing_hob(match_function):
    return high_order_bool.HighOrderBool(match_function.__name__, match_function)


@boxing_hob
def SameDeviceIds(context):
    x_blob_object, op_arg_parallel_attr = context
    blob_device_ids = x_blob_object.parallel_desc_symbol.machine_id2device_id_list
    arg_parallel_desc_symbol = op_arg_parallel_attr.parallel_desc_symbol
    op_device_ids = arg_parallel_desc_symbol.machine_id2device_id_list
    return blob_device_ids == op_device_ids


@boxing_hob
def SameSbpParallel(context):
    x_blob_object, op_arg_parallel_attr = context
    x_sbp_parallel = x_blob_object.op_arg_parallel_attr.sbp_parallel
    op_arg_sbp_parallel = op_arg_parallel_attr.sbp_parallel
    return x_sbp_parallel == op_arg_sbp_parallel


@boxing_hob
def HostToGpuDevice(context):
    x_blob_object, op_arg_parallel_attr = context
    blob_device_tag = x_blob_object.parallel_desc_symbol.device_tag
    op_device_tag = op_arg_parallel_attr.parallel_desc_symbol.device_tag
    return blob_device_tag == "cpu" and op_device_tag == "gpu"


@boxing_hob
def GpuDeviceToHost(context):
    x_blob_object, op_arg_parallel_attr = context
    blob_device_tag = x_blob_object.parallel_desc_symbol.device_tag
    op_device_tag = op_arg_parallel_attr.parallel_desc_symbol.device_tag
    return blob_device_tag == "gpu" and op_device_tag == "cpu"


@boxing_hob
def SameParallelDesc(context):
    x_blob_object, op_arg_parallel_attr = context
    x_parallel_desc_sym = x_blob_object.parallel_desc_symbol
    op_arg_parallel_desc_sym = op_arg_parallel_attr.parallel_desc_symbol
    return x_parallel_desc_sym == op_arg_parallel_desc_sym


@boxing_hob
def BlobOnSingleDevice(context):
    x_blob_object, op_arg_parallel_attr = context
    return x_blob_object.parallel_desc_symbol.parallel_num == 1


@boxing_hob
def SameMachineId(context):
    x_blob_object, op_arg_parallel_attr = context
    x_parallel_desc_sym = x_blob_object.parallel_desc_symbol
    return (
        x_parallel_desc_sym.machine_id2device_id_list.keys()
        == x_parallel_desc_sym.machine_id2device_id_list.keys()
    )


@boxing_hob
def OpArgParallelNumGt1(context):
    x_blob_object, op_arg_parallel_attr = context
    op_arg_parallel_desc_sym = op_arg_parallel_attr.parallel_desc_symbol
    return op_arg_parallel_desc_sym.parallel_num > 1


@boxing_hob
def OpArgSbpParallel(context):
    x_blob_object, op_arg_parallel_attr = context
    return op_arg_parallel_attr.sbp_parallel.HasField("broadcast_parallel")
