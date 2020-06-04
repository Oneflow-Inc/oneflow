from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow

def BuildCopHdInstruction(builder, x_blob_object):
    op_conf, lbi = _MakeCopyHdOpConfAndRetLbi()
    return _BuildCopyInstruction(builder, x_blob_object, op_conf)

def _MakeCopyHdOpConfAndRetLbi():
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = "copy_hd"
    op_conf.device_type = c_api_util.DeviceType4DeviceTag("gpu")
    setattr(op_conf.copy_conf, "in", "%s/in"%op_conf.name)
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
    bn_in_op2blob_object = {"in": x_blob_object}
    if current_device_tag == x_device_tag:
        builder.SystemStatelessCall(op_conf,
                const_arg_bns=["in"], mut_arg_bns=["out"], bn_in_op2blob_object=bn_in_op2blob_object)
    elif current_device_tag == "cpu" and x_device_tag == "gpu":
        x_parallel_conf = x_blob_object.parallel_desc_symbol.parallel_conf
        builder.SystemCudaD2HStatelessCall(op_conf, x_parallel_conf,
                const_arg_bns=["in"], mut_arg_bns=["out"], bn_in_op2blob_object=bn_in_op2blob_object)
    elif current_device_tag == "gpu" and x_device_tag == "cpu":
        out_parallel_conf = oneflow.placement.current_scope().default_parallel_conf
        with builder.CudaHostPinBlob(x_blob_object):
            builder.SystemCudaH2DStatelessCall(op_conf, out_parallel_conf,
                    const_arg_bns=["in"], mut_arg_bns=["out"], bn_in_op2blob_object=bn_in_op2blob_object)
    else:
        raise NotImplementedError("invalid device found. current_device_tag: %s, x_device_tag: %s"
                %(current_device_tag, x_device_tag))
    return bn_in_op2blob_object["out"]
