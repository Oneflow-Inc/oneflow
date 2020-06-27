from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.lib.core.high_order_bool as high_order_bool
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.job.placement_pb2 as placement_proto_pb
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.boxing_hob as boxing_hob
import oneflow


def BoxingTo(builder, x_blob_object, op_arg_parallel_attr):
    x_opt_mirrored_parallel = x_blob_object.op_arg_parallel_attr.opt_mirrored_parallel
    op_arg_opt_mirrored_parallel = op_arg_parallel_attr.opt_mirrored_parallel
    assert x_opt_mirrored_parallel == op_arg_opt_mirrored_parallel, (
        "\nx_arg_attribute: %s\nop_arg_parallel_attr: %s"
        % (x_blob_object.op_arg_parallel_attr, op_arg_parallel_attr)
    )
    conditional_functions = [SingleDeviceBoxing, BroadcastOneToMany, CopyHd]

    def default(get_failed_info, *args, **kwargs):
        raise NotImplementedError(
            "%s\n"
            "no boxing method found.\n"
            "logical_blob_name: %s\n"
            "x_arg_attribute: %s\n"
            "op_arg_parallel_attr: %s\n"
            % (
                get_failed_info(),
                x_blob_object.op_arg_blob_attr.logical_blob_name,
                x_blob_object.op_arg_parallel_attr,
                op_arg_parallel_attr,
            )
        )

    function = enable_if.unique(
        conditional_functions,
        context=(x_blob_object, op_arg_parallel_attr),
        default=default,
    )
    return function(builder, x_blob_object, op_arg_parallel_attr)


MatchCopyHd = (
    boxing_hob.SameDeviceIds
    & boxing_hob.SameSbpParallel
    & (boxing_hob.HostToGpuDevice | boxing_hob.GpuDeviceToHost)
)


@enable_if.condition(MatchCopyHd)
def CopyHd(builder, x_blob_object, op_arg_parallel_attr):
    arg_parallel_desc_symbol = op_arg_parallel_attr.parallel_desc_symbol
    op_device_tag = arg_parallel_desc_symbol.device_tag
    return BuildCopyHdInstruction(builder, x_blob_object, op_device_tag)


@enable_if.condition(boxing_hob.SameParallelDesc & boxing_hob.BlobOnSingleDevice)
def SingleDeviceBoxing(builder, x_blob_object, op_arg_parallel_attr):
    return x_blob_object


MatchBroadcastOneToMany = (
    boxing_hob.SameMachineId
    & boxing_hob.BlobOnSingleDevice
    & boxing_hob.OpArgParallelNumGt1
    & boxing_hob.OpArgSbpParallel
)


@enable_if.condition(MatchBroadcastOneToMany)
def BroadcastOneToMany(builder, x_blob_object, op_arg_parallel_attr):
    tmp_blob_object = OneToManyBroadcastBlobReference(
        builder, x_blob_object, op_arg_parallel_attr.parallel_desc_symbol
    )
    tmp_parallel_desc_sym = tmp_blob_object.parallel_desc_symbol
    op_arg_parallel_desc_sym = op_arg_parallel_attr.parallel_desc_symbol
    if tmp_parallel_desc_sym == op_arg_parallel_desc_sym:
        return tmp_blob_object
    ret = BuildCopyHdInstruction(
        builder, tmp_blob_object, op_arg_parallel_attr.parallel_desc_symbol.device_tag,
    )
    return ret


def Assign(builder, ref_blob_object, value_blob_object):
    return BuildAssignInstruction(
        builder, ref_blob_object, value_blob_object, _AssignOpConf()
    )


def OneToManyBroadcastBlobReference(builder, x_blob_object, to_parallel_desc_symbol):
    x_parallel_desc_symbol = x_blob_object.parallel_desc_symbol
    x_machine_ids = list(x_parallel_desc_symbol.machine_id2device_id_list.keys())
    to_machine_ids = list(to_parallel_desc_symbol.machine_id2device_id_list.keys())
    assert x_machine_ids == to_machine_ids, (x_machine_ids, to_machine_ids)
    x_first_device_ids = x_parallel_desc_symbol.machine_id2device_id_list[
        x_machine_ids[0]
    ]
    assert len(x_first_device_ids) == 1, x_first_device_ids
    if x_parallel_desc_symbol == to_parallel_desc_symbol:
        return x_blob_object
    tmp_parallel_desc_symbol = TryReplaceDeviceTag(
        builder, to_parallel_desc_symbol, x_parallel_desc_symbol.device_tag
    )
    ret = builder.BroadcastBlobReference(x_blob_object, tmp_parallel_desc_symbol)
    return ret


def BuildCopyHdInstruction(builder, x_blob_object, to_device_tag):
    op_conf, lbi = _MakeCopyHdOpConfAndRetLbi()
    return _BuildCopyInstruction(builder, x_blob_object, op_conf, to_device_tag)


def _MakeCopyHdOpConfAndRetLbi():
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = "copy_hd"
    op_conf.device_type = c_api_util.DeviceType4DeviceTag("gpu")
    setattr(op_conf.copy_conf, "in", "%s/in" % op_conf.name)
    op_conf.copy_conf.out = "out"
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return op_conf, lbi


def _BuildCopyInstruction(builder, x_blob_object, op_conf, to_device_tag):
    x_devices = x_blob_object.parallel_desc_symbol.machine_id2device_id_list
    x_device_tag = x_blob_object.parallel_desc_symbol.device_tag
    bn_in_op2blob_object = {"in": x_blob_object}
    op_attribute = c_api_util.GetOpAttribute4OpConf(op_conf)
    assert to_device_tag != x_device_tag, (to_device_tag, x_device_tag)
    if to_device_tag == "cpu" and x_device_tag == "gpu":
        x_parallel_conf = x_blob_object.parallel_desc_symbol.parallel_conf
        builder.BoxingCudaD2HStatelessCall(
            op_attribute, x_parallel_conf, bn_in_op2blob_object=bn_in_op2blob_object
        )
    elif to_device_tag == "gpu" and x_device_tag == "cpu":
        out_parallel_desc_symbol = TryReplaceDeviceTag(
            builder, x_blob_object.parallel_desc_symbol, to_device_tag
        )
        out_parallel_conf = out_parallel_desc_symbol.parallel_conf
        with builder.CudaHostPinBlob(x_blob_object):
            builder.BoxingCudaH2DStatelessCall(
                op_attribute,
                out_parallel_conf,
                bn_in_op2blob_object=bn_in_op2blob_object,
            )
    else:
        raise NotImplementedError(
            "invalid device found. to_device_tag: %s, x_device_tag: %s"
            % (to_device_tag, x_device_tag)
        )
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
    assert ref_devices == value_devices, "\nref_devices: %s\nvalue_devices: %s" % (
        ref_devices,
        value_devices,
    )
    ref_device_tag = ref_blob_object.parallel_desc_symbol.device_tag
    value_device_tag = value_blob_object.parallel_desc_symbol.device_tag
    bn_in_op2blob_object = {"ref": ref_blob_object, "value": value_blob_object}
    op_attribute = c_api_util.GetOpAttribute4OpConf(op_conf)
    if ref_device_tag == value_device_tag:
        builder.BoxingStatelessCall(
            op_attribute,
            parallel_conf=ref_parallel_conf,
            bn_in_op2blob_object=bn_in_op2blob_object,
        )
    elif ref_device_tag == "cpu" and value_device_tag == "gpu":
        value_parallel_conf = value_blob_object.parallel_desc_symbol.parallel_conf
        builder.BoxingCudaD2HStatelessCall(
            op_attribute, value_parallel_conf, bn_in_op2blob_object=bn_in_op2blob_object
        )
    elif ref_device_tag == "gpu" and value_device_tag == "cpu":
        with builder.CudaHostPinBlob(value_blob_object):
            builder.BoxingCudaH2DStatelessCall(
                op_attribute,
                ref_parallel_conf,
                bn_in_op2blob_object=bn_in_op2blob_object,
            )
    else:
        raise NotImplementedError(
            "invalid device found. ref_device_tag: %s, value_device_tag: %s"
            % (ref_device_tag, value_device_tag)
        )


def TryReplaceDeviceTag(builder, parallel_desc_symbol, device_tag):
    if parallel_desc_symbol.device_tag == device_tag:
        return parallel_desc_symbol
    parallel_conf = placement_proto_pb.ParallelConf()
    for device_name in parallel_desc_symbol.parallel_conf.device_name:
        triple = device_name.split(":")
        parallel_conf.device_name.append(
            "%s:%s:%s" % (triple[0], device_tag, triple[2])
        )
    return builder.GetParallelDescSymbol(parallel_conf)
