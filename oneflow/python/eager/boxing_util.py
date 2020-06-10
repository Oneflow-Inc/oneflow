from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.job.placement_pb2 as placement_proto_pb
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow

def Assign(builder, ref_blob_object, value_blob_object):
    return BuildAssignInstruction(builder, ref_blob_object, value_blob_object, _AssignOpConf())

def OneToManyBroadcastBlobReference(builder, x_blob_object, to_parallel_desc_symbol):
    x_parallel_desc_symbol = x_blob_object.parallel_desc_symbol
    x_machine_ids = list(x_parallel_desc_symbol.machine_id2device_id_list.keys())
    to_machine_ids = list(to_parallel_desc_symbol.machine_id2device_id_list.keys())
    assert x_machine_ids == to_machine_ids, (x_machine_ids, to_machine_ids)
    x_first_device_ids = x_parallel_desc_symbol.machine_id2device_id_list[x_machine_ids[0]]
    assert len(x_first_device_ids) == 1, x_first_device_ids
    if x_parallel_desc_symbol == to_parallel_desc_symbol: return x_blob_object
    tmp_parallel_desc_symbol = _TryReplaceDeviceTag(
            builder, to_parallel_desc_symbol, x_parallel_desc_symbol.device_tag)
    ret = builder.BroadcastBlobReference(x_blob_object, tmp_parallel_desc_symbol)
    return ret

def BuildCopyHdInstruction(builder, x_blob_object,
                          to_machine_id2device_id_list = None, to_device_tag = None):
    op_conf, lbi = _MakeCopyHdOpConfAndRetLbi()
    if to_machine_id2device_id_list is None:
        to_machine_id2device_id_list = oneflow.placement.current_scope().machine_id2device_id_list
    if to_device_tag is None: to_device_tag = oneflow.placement.current_scope().default_device_tag
    return _BuildCopyInstruction(builder, x_blob_object, op_conf,
            to_machine_id2device_id_list, to_device_tag)

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


def _BuildCopyInstruction(builder, x_blob_object, op_conf, current_devices, current_device_tag):
    x_devices = x_blob_object.parallel_desc_symbol.machine_id2device_id_list
    assert current_devices == x_devices,\
            "\ncurrent_devices: %s\nx_devices: %s" %(current_devices, x_devices)
    x_device_tag = x_blob_object.parallel_desc_symbol.device_tag
    bn_in_op2blob_object = {"in": x_blob_object}
    op_attribute = c_api_util.GetOpAttribute4OpConf(op_conf)
    if current_device_tag == x_device_tag:
        bn_in_op2blob_object['out'] = x_blob_object
    elif current_device_tag == "cpu" and x_device_tag == "gpu":
        x_parallel_conf = x_blob_object.parallel_desc_symbol.parallel_conf
        builder.CudaD2HStatelessCall(op_attribute, x_parallel_conf,
                bn_in_op2blob_object=bn_in_op2blob_object)
    elif current_device_tag == "gpu" and x_device_tag == "cpu":
        out_parallel_conf = oneflow.placement.current_scope().default_parallel_conf
        with builder.CudaHostPinBlob(x_blob_object):
            builder.CudaH2DStatelessCall(op_attribute, out_parallel_conf,
                    bn_in_op2blob_object=bn_in_op2blob_object)
    else:
        raise NotImplementedError("invalid device found. current_device_tag: %s, x_device_tag: %s"
                %(current_device_tag, x_device_tag))
    return bn_in_op2blob_object["out"]

def _AssignOpConf():
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = "assign"
    op_conf.assign_conf.ref = "assign/ref"
    op_conf.assign_conf.value = "assign/value"
    return op_conf

def BuildAssignInstruction(builder, ref_blob_object, value_blob_object, op_conf):
    blob_cache_util.TryDisableBlobCache(ref_blob_object)
    ref_parallel_conf = ref_blob_object.parallel_desc_symbol.parallel_conf
    ref_devices = ref_blob_object.parallel_desc_symbol.machine_id2device_id_list
    value_devices = value_blob_object.parallel_desc_symbol.machine_id2device_id_list
    assert ref_devices == value_devices,\
            "\nref_devices: %s\nvalue_devices: %s" %(ref_devices, value_devices)
    ref_device_tag = ref_blob_object.parallel_desc_symbol.device_tag
    value_device_tag = value_blob_object.parallel_desc_symbol.device_tag
    bn_in_op2blob_object = {"ref": ref_blob_object, "value": value_blob_object}
    op_attribute = c_api_util.GetOpAttribute4OpConf(op_conf)
    if ref_device_tag == value_device_tag:
        builder.StatelessCall(op_attribute,
                parallel_conf=ref_parallel_conf, device_tag=ref_device_tag,
                bn_in_op2blob_object=bn_in_op2blob_object)
    elif ref_device_tag == "cpu" and value_device_tag == "gpu":
        value_parallel_conf = value_blob_object.parallel_desc_symbol.parallel_conf
        builder.CudaD2HStatelessCall(op_attribute, value_parallel_conf,
                bn_in_op2blob_object=bn_in_op2blob_object)
    elif ref_device_tag == "gpu" and value_device_tag == "cpu":
        with builder.CudaHostPinBlob(value_blob_object):
            builder.CudaH2DStatelessCall(op_attribute, ref_parallel_conf,
                    bn_in_op2blob_object=bn_in_op2blob_object)
    else:
        raise NotImplementedError("invalid device found. ref_device_tag: %s, value_device_tag: %s"
                              %(ref_device_tag, value_device_tag))

def _TryReplaceDeviceTag(builder, parallel_desc_symbol, device_tag):
    if parallel_desc_symbol.device_tag == device_tag: return parallel_desc_symbol
    parallel_conf = placement_proto_pb.ParallelConf()
    for device_name in parallel_desc_symbol.parallel_conf.device_name:
        triple = device_name.split(':')
        parallel_conf.device_name.append("%s:%s:%s"%(triple[0], device_tag, triple[2]))
    return builder.GetParallelDescSymbol(parallel_conf, device_tag)
