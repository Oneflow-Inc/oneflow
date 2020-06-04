from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow

def BuildCopHdInstruction(builder, x_lbn, x_blob_object):
    op_conf, lbi = _MakeCopyHdOpConfAndRetLbi(x_lbn)
    _BuildCopyInstruction(builder, x_blob_object, op_conf)
    return "%s/%s"%(lbi.op_name, lbi.blob_name)

def _MakeCopyHdOpConfAndRetLbi(x_lbn):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("Copy_")
    op_conf.device_type = c_api_util.DeviceType4DeviceTag("gpu")
    setattr(op_conf.copy_conf, "in", x_lbn)
    op_conf.copy_conf.out = "out"
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return op_conf, lbi


def _BuildCopyInstruction(builder, x_blob_object, op_conf):
    current_devices = oneflow.placement.current_scope().machine_id2device_id_list
    x_devices = x_blob_object.parallel_desc_symbol.machine_id2device_id_list
    assert current_devices == x_devices,\
            "\ncurrent_devices: %s\nx_devices: %s" %(current_devices, x_devices)
    current_device_tag = oneflow.placement.current_scope().default_device_tag
    x_device_tag = x_blob_object.parallel_desc_symbol.device_tag
    if current_device_tag == x_device_tag:
        builder.SystemStatelessCall(op_conf,
                const_arg_bns=["in"], mut_arg_bns=["out"])
    elif current_device_tag == "cpu" and x_device_tag == "gpu":
        x_parallel_conf = x_blob_object.parallel_desc_symbol.parallel_conf
        builder.SystemCudaD2HStatelessCall(op_conf, x_parallel_conf,
                const_arg_bns=["in"], mut_arg_bns=["out"])
    elif current_device_tag == "gpu" and x_device_tag == "cpu":
        out_parallel_conf = oneflow.placement.current_scope().default_parallel_conf
        with builder.CudaHostPinBlob(x_blob_object):
            builder.SystemCudaH2DStatelessCall(op_conf, out_parallel_conf,
                    const_arg_bns=["in"], mut_arg_bns=["out"])
    else:
        raise NotImplementedError("invalid device found. current_device_tag: %s, x_device_tag: %s"
                %(current_device_tag, x_device_tag))

