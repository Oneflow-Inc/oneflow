from __future__ import absolute_import

import oneflow.python.eager.symbol as symbol_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.job.sbp_parallel_pb2 as sbp_parallel_pb
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.op_arg_util as op_arg_util
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.lib.core.high_order_bool as high_order_bool
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.job.placement_pb2 as placement_proto_pb
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.boxing_hob as boxing_hob
import oneflow


def BoxingTo(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    hob_context = (produced_blob_object, consumer_op_arg_parallel_attr)
    if enable_if.get_condition_hob(NoBoxing)(hob_context):
        return produced_blob_object

    producer_opt_mirrored_parallel = (
        produced_blob_object.op_arg_parallel_attr.opt_mirrored_parallel
    )
    consumer_opt_mirrored_parallel = consumer_op_arg_parallel_attr.opt_mirrored_parallel
    assert producer_opt_mirrored_parallel == consumer_opt_mirrored_parallel, (
        "\nproducer_op_arg_parallel_attr: %s\nconsumer_op_arg_parallel_attr: %s"
        % (produced_blob_object.op_arg_parallel_attr, consumer_op_arg_parallel_attr)
    )

    def default(get_failed_info, *args, **kwargs):
        raise NotImplementedError(
            "%s\n"
            "no boxing method found.\n"
            "logical_blob_name: %s\n"
            "x_arg_attribute: %s\n"
            "consumer_op_arg_parallel_attr: %s\n"
            % (
                get_failed_info(),
                produced_blob_object.op_arg_blob_attr.logical_blob_name,
                produced_blob_object.op_arg_parallel_attr,
                consumer_op_arg_parallel_attr,
            )
        )

    conditional_functions = [
        CopyH2D,
        CopyD2H,
        NoBoxing,
        # e.g. 0:cpu:0 -> 0:cpu:0-3
        BroadcastOneToMany,
        # e.g. 0:cpu:0 -> 0:gpu:0-3
        # ==>   0:cpu:0 -> 0:cpu:0-3 -> 0:gpu:0-3
        Sequential((BroadcastOneToMany, BroadcastBlobParallelDesc("cpu")), CopyH2D),
        Sequential((CopyD2H, ReplaceBlobDeviceTag("cpu")), BroadcastOneToMany),
        BroadcastManyToOne,
        Sequential((BroadcastManyToOne, ReplaceBlobParallelDesc("cpu")), CopyH2D),
        Sequential((BroadcastManyToOne, ReplaceBlobParallelDesc("gpu")), CopyD2H),
        # e.g. 0:gpu:0-3 -> 0:gpu:0-3 (P->B)
        NcclAllReduce,
        Sequential((CopyH2D, ReplaceBlobParallelDesc("gpu")), NcclAllReduce),
        Sequential((NcclAllReduce, GetBroadcastOpArgParallelAttr), CopyD2H),
    ]
    function = enable_if.unique(
        conditional_functions,
        context=(produced_blob_object, consumer_op_arg_parallel_attr),
        default=default,
    )
    return function(builder, produced_blob_object, consumer_op_arg_parallel_attr)


def ComposeBoxing(lhs_boxing, rhs_boxing, get_medium_op_arg_parallel_attr):
    composed_hob = boxing_hob.ComposeHob(
        enable_if.get_condition_hob(lhs_boxing),
        enable_if.get_condition_hob(rhs_boxing),
        get_medium_op_arg_parallel_attr=get_medium_op_arg_parallel_attr,
    )

    @enable_if.condition(composed_hob)
    def Composed(builder, produced_blob_object, consumer_op_arg_parallel_attr):
        tmp_op_arg_parallel_attr = get_medium_op_arg_parallel_attr(
            builder, produced_blob_object, consumer_op_arg_parallel_attr
        )
        tmp = lhs_boxing(builder, produced_blob_object, tmp_op_arg_parallel_attr)
        return rhs_boxing(builder, tmp, consumer_op_arg_parallel_attr)

    Composed.__debug_str__ = "%s->%s" % (
        lhs_boxing.__debug_str__
        if hasattr(lhs_boxing, "__debug_str__")
        else lhs_boxing.__name__,
        rhs_boxing.__debug_str__
        if hasattr(rhs_boxing, "__debug_str__")
        else rhs_boxing.__name__,
    )
    return Composed


def Sequential(*boxing_methods):
    def GetBoxingMethodWithDefault(boxing_method):
        if isinstance(boxing_method, tuple):
            assert len(boxing_method) == 2
            return boxing_method
        else:

            def Default(_0, produced_blob_object, _):
                return produced_blob_object.op_arg_parallel_attr

            return boxing_method, Default

    assert not isinstance(boxing_methods[-1], tuple)
    composed = boxing_methods[-1]
    for pair in boxing_methods[-2::-1]:
        boxing_method, medium_getter = GetBoxingMethodWithDefault(pair)
        composed = ComposeBoxing(
            boxing_method, composed, get_medium_op_arg_parallel_attr=medium_getter
        )
    return composed


MatchCopyH2D = (
    boxing_hob.MasterMachineOnly
    & (
        boxing_hob.producer_parallel_desc.machine_id2device_id_list
        == boxing_hob.consumer_parallel_desc.machine_id2device_id_list
    )
    & (
        (boxing_hob.producer_sbp_parallel == boxing_hob.consumer_sbp_parallel)
        | (boxing_hob.producer_parallel_desc.parallel_num == 1)
    )
    & (boxing_hob.producer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.consumer_parallel_desc.device_tag == "gpu")
)


@enable_if.condition(MatchCopyH2D)
def CopyH2D(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return CopyHD(builder, produced_blob_object, consumer_op_arg_parallel_attr)


MatchCopyD2H = (
    boxing_hob.MasterMachineOnly
    & (
        boxing_hob.producer_parallel_desc.machine_id2device_id_list
        == boxing_hob.consumer_parallel_desc.machine_id2device_id_list
    )
    & (
        (boxing_hob.producer_sbp_parallel == boxing_hob.consumer_sbp_parallel)
        | (boxing_hob.producer_parallel_desc.parallel_num == 1)
    )
    & (boxing_hob.producer_parallel_desc.device_tag == "gpu")
    & (boxing_hob.consumer_parallel_desc.device_tag == "cpu")
)


@enable_if.condition(MatchCopyD2H)
def CopyD2H(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return CopyHD(builder, produced_blob_object, consumer_op_arg_parallel_attr)


def CopyHD(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    arg_parallel_desc_symbol = consumer_op_arg_parallel_attr.parallel_desc_symbol
    op_device_tag = arg_parallel_desc_symbol.device_tag
    return BuildCopyHdInstruction(builder, produced_blob_object, op_device_tag)


MatchNoBoxing = (
    boxing_hob.MasterMachineOnly
    & (boxing_hob.producer_parallel_desc == boxing_hob.consumer_parallel_desc)
    & (
        (boxing_hob.producer_sbp_parallel == boxing_hob.consumer_sbp_parallel)
        | (boxing_hob.producer_parallel_desc.parallel_num == 1)
    )
)


@enable_if.condition(MatchNoBoxing)
def NoBoxing(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return produced_blob_object


BlobIsPartialSum = boxing_hob.producer_sbp_parallel.HasField("partial_sum_parallel")
OpArgIsBroadcast = boxing_hob.consumer_sbp_parallel.HasField("broadcast_parallel")


MatchNcclAllReduce = (
    boxing_hob.MasterMachineOnly
    & (boxing_hob.producer_parallel_desc.device_tag == "gpu")
    & (boxing_hob.producer_parallel_desc == boxing_hob.consumer_parallel_desc)
    & (boxing_hob.consumer_parallel_desc.parallel_num > 1)
    & BlobIsPartialSum
    & OpArgIsBroadcast
)


@enable_if.condition(MatchNcclAllReduce)
def NcclAllReduce(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    parallel_conf = consumer_op_arg_parallel_attr.parallel_desc_symbol.parallel_conf
    op_attribute = _GetEagerNcclAllReduce(parallel_conf)
    bn_in_op2blob_object = dict(in_0=produced_blob_object)
    builder.BoxingStatelessCall(
        op_attribute,
        parallel_conf=parallel_conf,
        bn_in_op2blob_object=bn_in_op2blob_object,
    )
    y_blob_object = bn_in_op2blob_object["out_0"]
    y_blob_object.op_arg_parallel_attr.Assign(consumer_op_arg_parallel_attr)
    return y_blob_object


def ReplaceBlobParallelDesc(new_device_tag):
    def GetOpArgParallelAttr(builder, produced_blob_object, op_arg_parallal_attr):
        x_parallel_attr = produced_blob_object.op_arg_parallel_attr
        parallel_desc_sym = op_arg_parallal_attr.parallel_desc_symbol
        new_parallel_desc_symbol = TryReplaceDeviceTag(
            builder, parallel_desc_sym, new_device_tag
        )
        return op_arg_util.OpArgParallelAttribute(
            new_parallel_desc_symbol,
            x_parallel_attr.sbp_parallel,
            x_parallel_attr.opt_mirrored_parallel,
        )

    return GetOpArgParallelAttr


def BroadcastBlobParallelDesc(new_device_tag):
    def GetOpArgParallelAttr(builder, produced_blob_object, op_arg_parallal_attr):
        x_parallel_attr = produced_blob_object.op_arg_parallel_attr
        parallel_desc_sym = op_arg_parallal_attr.parallel_desc_symbol
        new_parallel_desc_symbol = TryReplaceDeviceTag(
            builder, parallel_desc_sym, new_device_tag
        )
        sbp_parallel = sbp_parallel_pb.SbpParallel()
        sbp_parallel.broadcast_parallel.SetInParent()
        return op_arg_util.OpArgParallelAttribute(
            new_parallel_desc_symbol,
            sbp_parallel,
            x_parallel_attr.opt_mirrored_parallel,
        )

    return GetOpArgParallelAttr


def ReplaceBlobDeviceTag(new_device_tag):
    def GetOpArgParallelAttr(builder, produced_blob_object, op_arg_parallal_attr):
        x_parallel_attr = produced_blob_object.op_arg_parallel_attr
        new_parallel_desc_symbol = TryReplaceDeviceTag(
            builder, x_parallel_attr.parallel_desc_symbol, new_device_tag
        )
        return op_arg_util.OpArgParallelAttribute(
            new_parallel_desc_symbol,
            x_parallel_attr.sbp_parallel,
            x_parallel_attr.opt_mirrored_parallel,
        )

    return GetOpArgParallelAttr


def GetBroadcastOpArgParallelAttr(builder, produced_blob_object, op_arg_parallal_attr):
    x_parallel_attr = produced_blob_object.op_arg_parallel_attr
    sbp_parallel = sbp_parallel_pb.SbpParallel()
    sbp_parallel.broadcast_parallel.SetInParent()
    return op_arg_util.OpArgParallelAttribute(
        x_parallel_attr.parallel_desc_symbol,
        sbp_parallel,
        x_parallel_attr.opt_mirrored_parallel,
    )


MatchBroadcastOneToMany = (
    boxing_hob.MasterMachineOnly
    & boxing_hob.ProducerDevicesContainedInConsumerDevices
    & (boxing_hob.producer_parallel_desc.parallel_num == 1)
    & (boxing_hob.consumer_parallel_desc.parallel_num > 1)
    & boxing_hob.consumer_sbp_parallel.HasField("broadcast_parallel")
)


@enable_if.condition(MatchBroadcastOneToMany)
def BroadcastOneToMany(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    tmp_blob_object = OneToManyBroadcastBlobReference(
        builder,
        produced_blob_object,
        consumer_op_arg_parallel_attr.parallel_desc_symbol,
    )
    tmp_parallel_desc_sym = tmp_blob_object.parallel_desc_symbol
    op_arg_parallel_desc_sym = consumer_op_arg_parallel_attr.parallel_desc_symbol
    if tmp_parallel_desc_sym == op_arg_parallel_desc_sym:
        return tmp_blob_object
    ret = BuildCopyHdInstruction(
        builder,
        tmp_blob_object,
        consumer_op_arg_parallel_attr.parallel_desc_symbol.device_tag,
    )
    return ret


MatchBroadcastManyToOne = (
    boxing_hob.MasterMachineOnly
    & boxing_hob.ConsumerDevicesContainedInProducerDevices
    & (boxing_hob.producer_parallel_desc.parallel_num > 1)
    & (boxing_hob.consumer_parallel_desc.parallel_num == 1)
    & boxing_hob.producer_sbp_parallel.HasField("broadcast_parallel")
)


@enable_if.condition(MatchBroadcastManyToOne)
def BroadcastManyToOne(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    y_blob_objects = builder.UnpackLogicalBlobToPhysicalBlobs(produced_blob_object)
    for y in y_blob_objects:
        if y.parallel_desc_symbol == consumer_op_arg_parallel_attr.parallel_desc_symbol:
            return y
    raise NotImplementedError("op_arg's devices is not contained in blob's devices")


def Assign(builder, ref_blob_object, value_blob_object):
    return BuildAssignInstruction(
        builder, ref_blob_object, value_blob_object, _AssignOpConf()
    )


def OneToManyBroadcastBlobReference(
    builder, produced_blob_object, to_parallel_desc_symbol
):
    x_parallel_desc_symbol = produced_blob_object.parallel_desc_symbol
    x_machine_ids = list(x_parallel_desc_symbol.machine_id2device_id_list.keys())
    to_machine_ids = list(to_parallel_desc_symbol.machine_id2device_id_list.keys())
    assert x_machine_ids == to_machine_ids, (x_machine_ids, to_machine_ids)
    x_first_device_ids = x_parallel_desc_symbol.machine_id2device_id_list[
        x_machine_ids[0]
    ]
    assert len(x_first_device_ids) == 1, x_first_device_ids
    if x_parallel_desc_symbol == to_parallel_desc_symbol:
        return produced_blob_object
    tmp_parallel_desc_symbol = TryReplaceDeviceTag(
        builder, to_parallel_desc_symbol, x_parallel_desc_symbol.device_tag
    )
    ret = builder.BroadcastBlobReference(produced_blob_object, tmp_parallel_desc_symbol)
    return ret


def BuildCopyHdInstruction(builder, produced_blob_object, to_device_tag):
    op_conf, lbi = _MakeCopyHdOpConfAndRetLbi()
    return _BuildCopyInstruction(builder, produced_blob_object, op_conf, to_device_tag)


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


def _BuildCopyInstruction(builder, produced_blob_object, op_conf, to_device_tag):
    x_devices = produced_blob_object.parallel_desc_symbol.machine_id2device_id_list
    x_device_tag = produced_blob_object.parallel_desc_symbol.device_tag
    bn_in_op2blob_object = {"in": produced_blob_object}
    op_attribute = c_api_util.GetOpAttribute4OpConf(op_conf)
    assert to_device_tag != x_device_tag, (to_device_tag, x_device_tag)
    if to_device_tag == "cpu" and x_device_tag == "gpu":
        x_parallel_conf = produced_blob_object.parallel_desc_symbol.parallel_conf
        builder.BoxingCudaD2HStatelessCall(
            op_attribute, x_parallel_conf, bn_in_op2blob_object=bn_in_op2blob_object
        )
    elif to_device_tag == "gpu" and x_device_tag == "cpu":
        out_parallel_desc_symbol = TryReplaceDeviceTag(
            builder, produced_blob_object.parallel_desc_symbol, to_device_tag
        )
        out_parallel_conf = out_parallel_desc_symbol.parallel_conf
        with builder.CudaHostPinBlob(produced_blob_object):
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
    sbp_parallel = bn_in_op2blob_object["out"].op_arg_parallel_attr.sbp_parallel
    sbp_parallel.CopyFrom(produced_blob_object.op_arg_parallel_attr.sbp_parallel)
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
    else:
        return ReplaceDeviceTag(parallel_desc_symbol, device_tag, builder=builder)


def ReplaceDeviceTag(parallel_desc_symbol, device_tag, builder=None):
    assert parallel_desc_symbol.device_tag != device_tag
    parallel_conf = placement_proto_pb.ParallelConf()
    for device_name in parallel_desc_symbol.parallel_conf.device_name:
        triple = device_name.split(":")
        parallel_conf.device_name.append(
            "%s:%s:%s" % (triple[0], device_tag, triple[2])
        )
    if builder is None:
        return symbol_util.ParallelDescSymbol(
            parallel_desc_symbol.symbol_id, parallel_conf, device_tag
        )
    else:
        return builder.GetParallelDescSymbol(parallel_conf)


def _GetEagerNcclAllReduce(parallel_conf):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = "eager_nccl_all_reduce"
    op_conf.user_conf.op_type_name = "eager_nccl_all_reduce"
    op_conf.user_conf.input["in"].s.append("eager_nccl_all_reduce/in_0")
    op_conf.user_conf.output["out"].s.append("eager_nccl_all_reduce/out_0")
    op_conf.user_conf.attr["parallel_conf"].at_string = str(parallel_conf)
    return c_api_util.GetOpAttribute4OpConf(op_conf)
