from __future__ import absolute_import

import oneflow.python.eager.symbol as symbol_util
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.core.job.sbp_parallel_pb2 as sbp_parallel_pb
import oneflow.core.job.placement_pb2 as placement_pb
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.op_arg_util as op_arg_util
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.lib.core.high_order_bool as high_order_bool
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.job.placement_pb2 as placement_pb
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.boxing_hob as boxing_hob
import oneflow.python.eager.boxing_middle as boxing_middle
import random
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
        CpuManyOneToOne,
        Sequential(
            boxing_middle.BoxingToMiddle(
                CopyD2H,
                boxing_middle.ReplaceProducerDeviceTag("cpu"),
                boxing_middle.ProducerSbpParallel,
            ),
            CpuManyOneToOne,
        ),
        Sequential(
            boxing_middle.BoxingToMiddle(
                CpuManyOneToOne,
                boxing_middle.ReplaceConsumerDeviceTag("cpu"),
                boxing_middle.ConsumerSbpParallel,
            ),
            CopyH2D,
        ),
        Sequential(
            boxing_middle.BoxingToMiddle(
                CopyD2H,
                boxing_middle.ReplaceProducerDeviceTag("cpu"),
                boxing_middle.ProducerSbpParallel,
            ),
            boxing_middle.BoxingToMiddle(
                CpuManyOneToOne,
                boxing_middle.ReplaceConsumerDeviceTag("cpu"),
                boxing_middle.ConsumerSbpParallel,
            ),
            CopyH2D,
        ),
        # e.g. 0:cpu:0 -> 0:cpu:0-3
        BroadcastOneToMany,
        # e.g. 0:cpu:0 -> 0:gpu:0-3
        # ==>   0:cpu:0 -> 0:cpu:0-3 -> 0:gpu:0-3
        Sequential(
            boxing_middle.BoxingToMiddle(
                BroadcastOneToMany,
                boxing_middle.ReplaceConsumerDeviceTag("cpu"),
                boxing_middle.BroadcastParallel,
            ),
            CopyH2D,
        ),
        Sequential(
            boxing_middle.BoxingToMiddle(
                CopyD2H,
                boxing_middle.ReplaceProducerDeviceTag("cpu"),
                boxing_middle.ProducerSbpParallel,
            ),
            BroadcastOneToMany,
        ),
        BroadcastManyToOne,
        Sequential(
            boxing_middle.BoxingToMiddle(
                BroadcastManyToOne,
                boxing_middle.ReplaceConsumerDeviceTag("cpu"),
                boxing_middle.ProducerSbpParallel,
            ),
            CopyH2D,
        ),
        Sequential(
            boxing_middle.BoxingToMiddle(
                BroadcastManyToOne,
                boxing_middle.ReplaceConsumerDeviceTag("gpu"),
                boxing_middle.ProducerSbpParallel,
            ),
            CopyD2H,
        ),
        # e.g. 0:gpu:0-3 -> 0:gpu:0-3 (P->B)
        NcclAllReduce,
        Sequential(
            boxing_middle.BoxingToMiddle(
                CopyH2D,
                boxing_middle.ReplaceConsumerDeviceTag("gpu"),
                boxing_middle.ProducerSbpParallel,
            ),
            NcclAllReduce,
        ),
        Sequential(
            boxing_middle.BoxingToMiddle(
                NcclAllReduce,
                boxing_middle.ProducerParallelDesc,
                boxing_middle.BroadcastParallel,
            ),
            CopyD2H,
        ),
        NaiveCpuConcatSplit,
        Sequential(
            boxing_middle.BoxingToMiddle(
                CopyD2H,
                boxing_middle.ReplaceProducerDeviceTag("cpu"),
                boxing_middle.ProducerSbpParallel,
            ),
            NaiveCpuConcatSplit,
        ),
        Sequential(
            boxing_middle.BoxingToMiddle(
                NaiveCpuConcatSplit,
                boxing_middle.ReplaceConsumerDeviceTag("cpu"),
                boxing_middle.ConsumerSbpParallel,
            ),
            CopyH2D,
        ),
        Sequential(
            boxing_middle.BoxingToMiddle(
                CopyD2H,
                boxing_middle.ReplaceProducerDeviceTag("cpu"),
                boxing_middle.ProducerSbpParallel,
            ),
            boxing_middle.BoxingToMiddle(
                NaiveCpuConcatSplit,
                boxing_middle.ReplaceConsumerDeviceTag("cpu"),
                boxing_middle.ConsumerSbpParallel,
            ),
            CopyH2D,
        ),
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
    assert not isinstance(boxing_methods[-1], boxing_middle.BoxingToMiddle)
    composed = boxing_methods[-1]
    for boxing_to_middle in boxing_methods[-2::-1]:
        assert isinstance(boxing_to_middle, boxing_middle.BoxingToMiddle)
        composed = ComposeBoxing(
            boxing_to_middle.boxing_method,
            composed,
            boxing_to_middle.get_middle_op_arg_parallel_attr,
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


MatchCpuManyOneToOne = (
    boxing_hob.MasterMachineOnly
    & (boxing_hob.producer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.consumer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.producer_parallel_desc != boxing_hob.consumer_parallel_desc)
    & (
        boxing_hob.producer_parallel_desc.parallel_num
        == boxing_hob.consumer_parallel_desc.parallel_num
    )
    & (boxing_hob.producer_parallel_desc.parallel_num > 1)
    & (boxing_hob.producer_sbp_parallel == boxing_hob.consumer_sbp_parallel)
)


@enable_if.condition(MatchCpuManyOneToOne)
def CpuManyOneToOne(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    def get_identity_physical_in_blob_objects(
        builder,
        produced_blob_object,
        consumer_op_arg_parallel_attr,
        physical_in_blob_objects,
        boxing_parallel_desc_symbol,
        out_parallel_num,
    ):
        return physical_in_blob_objects

    return NaiveCpuRefPhysicalBlobObjectsScope(
        builder,
        produced_blob_object,
        consumer_op_arg_parallel_attr,
        get_physical_out_blob_objects=get_identity_physical_in_blob_objects,
    )


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


MatchSplitOneToMany = (
    (boxing_hob.producer_parallel_desc.parallel_num == 1)
    & (boxing_hob.consumer_parallel_desc.parallel_num > 1)
    & boxing_hob.consumer_sbp_parallel.HasField("split_parallel")
)

MatchConcatManyToOne = (
    (boxing_hob.consumer_parallel_desc.parallel_num == 1)
    & (boxing_hob.producer_parallel_desc.parallel_num > 1)
    & boxing_hob.producer_sbp_parallel.HasField("split_parallel")
)

MatchConcatManyToSplitMany = (
    (boxing_hob.producer_parallel_desc.parallel_num > 1)
    & (boxing_hob.consumer_parallel_desc.parallel_num > 1)
    & boxing_hob.producer_sbp_parallel.HasField("split_parallel")
    & boxing_hob.consumer_sbp_parallel.HasField("split_parallel")
    & (
        (boxing_hob.producer_sbp_parallel != boxing_hob.consumer_sbp_parallel)
        | (
            boxing_hob.producer_parallel_desc.parallel_num
            != boxing_hob.consumer_parallel_desc.parallel_num
        )
    )
)


MatchNaiveCpuConcatSplit = (
    boxing_hob.MasterMachineOnly
    & (boxing_hob.producer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.consumer_parallel_desc.device_tag == "cpu")
    & (MatchSplitOneToMany | MatchConcatManyToOne | MatchConcatManyToSplitMany)
)


@enable_if.condition(MatchNaiveCpuConcatSplit)
def NaiveCpuConcatSplit(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return NaiveCpuRefPhysicalBlobObjectsScope(
        builder,
        produced_blob_object,
        consumer_op_arg_parallel_attr,
        get_physical_out_blob_objects=NaiveBoxingToPhysicalBlobObjects,
    )


def NaiveCpuRefPhysicalBlobObjectsScope(
    builder,
    produced_blob_object,
    consumer_op_arg_parallel_attr,
    get_physical_out_blob_objects,
):
    physical_in_blob_objects = UnpackLogicalBoxingBlobObjectToPhysical(
        builder, produced_blob_object
    )
    out_parallel_num = consumer_op_arg_parallel_attr.parallel_desc_symbol.parallel_num
    boxing_parallel_desc_symbol = GetConcatSplitBoxingParallelDescSymbol(
        builder,
        produced_blob_object.parallel_desc_symbol,
        max(len(physical_in_blob_objects), out_parallel_num),
    )
    physical_in_blob_objects = RefBlobObjectWithParallelDesc(
        builder,
        physical_in_blob_objects,
        [boxing_parallel_desc_symbol] * len(physical_in_blob_objects),
    )
    physical_output_blob_objects = get_physical_out_blob_objects(
        builder=builder,
        produced_blob_object=produced_blob_object,
        consumer_op_arg_parallel_attr=consumer_op_arg_parallel_attr,
        physical_in_blob_objects=physical_in_blob_objects,
        boxing_parallel_desc_symbol=boxing_parallel_desc_symbol,
        out_parallel_num=out_parallel_num,
    )
    phy_parallel_desc_symbols = builder.GetPhysicalParallelDescSymbols(
        consumer_op_arg_parallel_attr.parallel_desc_symbol
    )
    physical_output_blob_objects = RefBlobObjectWithParallelDesc(
        builder, physical_output_blob_objects, phy_parallel_desc_symbols
    )
    return PackPhysicalBoxingBlobObjectsToLogical(
        builder,
        physical_output_blob_objects,
        consumer_op_arg_parallel_attr,
        produced_blob_object.op_arg_blob_attr,
    )


def NaiveBoxingToPhysicalBlobObjects(
    builder,
    produced_blob_object,
    consumer_op_arg_parallel_attr,
    physical_in_blob_objects,
    boxing_parallel_desc_symbol,
    out_parallel_num,
):
    op_attribute = ConstructConcatSplitBoxingOpConf(
        produced_blob_object,
        consumer_op_arg_parallel_attr,
        len(physical_in_blob_objects),
        out_parallel_num,
    )
    return BuildConcatSplitBoxing(
        builder,
        op_attribute,
        physical_in_blob_objects,
        boxing_parallel_desc_symbol,
        out_parallel_num,
    )


def RefBlobObjectWithParallelDesc(
    builder, physical_blob_objects, phy_parallel_desc_symbols
):
    assert len(physical_blob_objects) == len(
        phy_parallel_desc_symbols
    ), "%s v.s. %s" % (len(physical_blob_objects), len(phy_parallel_desc_symbols))

    def RefWithParallelDesc(physical_blob_object, phy_parallel_desc_symbol):
        if physical_blob_object.parallel_desc_symbol == phy_parallel_desc_symbol:
            return physical_blob_object
        return builder.BroadcastBlobReference(
            physical_blob_object, phy_parallel_desc_symbol
        )

    return [
        RefWithParallelDesc(*pair)
        for pair in zip(physical_blob_objects, phy_parallel_desc_symbols)
    ]


def PackPhysicalBoxingBlobObjectsToLogical(
    builder, physical_blob_objects, op_arg_parallel_attr, op_arg_blob_attr
):
    if len(physical_blob_objects) == 1:
        return physical_blob_objects[0]
    return builder.PackPhysicalBlobsToLogicalBlob(
        physical_blob_objects, op_arg_parallel_attr, op_arg_blob_attr
    )


def BuildConcatSplitBoxing(
    builder,
    op_attribute,
    physical_in_blob_objects,
    boxing_parallel_desc_symbol,
    out_parallel_num,
):
    bn_in_op2blob_object = {}
    for i in range(len(physical_in_blob_objects)):
        bn_in_op2blob_object["in_%s" % i] = physical_in_blob_objects[i]
    builder.BoxingStatelessCall(
        op_attribute,
        parallel_conf=boxing_parallel_desc_symbol.parallel_conf,
        bn_in_op2blob_object=bn_in_op2blob_object,
    )
    return [bn_in_op2blob_object["out_%s" % i] for i in range(out_parallel_num)]


def ConstructConcatSplitBoxingOpConf(
    produced_blob_object,
    consumer_op_arg_parallel_attr,
    in_parallel_num,
    out_parallel_num,
):
    op_conf = op_conf_pb.OperatorConf()
    op_conf.name = "undefined_boxing_op_name"
    op_conf.boxing_conf.lbi.op_name = "undefined_boxing_op_name"
    op_conf.boxing_conf.lbi.blob_name = "undefined_boxing_blob_name"
    op_conf.boxing_conf.in_num = in_parallel_num
    op_conf.boxing_conf.out_num = out_parallel_num
    in_sbp_parallel = produced_blob_object.op_arg_parallel_attr.sbp_parallel
    if in_sbp_parallel.HasField("split_parallel"):
        in_axis = in_sbp_parallel.split_parallel.axis
    else:
        assert in_parallel_num == 1
        in_axis = 0
    op_conf.boxing_conf.concat_box.axis = in_axis
    out_sbp_parallel = consumer_op_arg_parallel_attr.sbp_parallel
    if out_sbp_parallel.HasField("split_parallel"):
        out_axis = out_sbp_parallel.split_parallel.axis
    else:
        assert out_parallel_num == 1
        out_axis = 0
    op_conf.boxing_conf.split_box.axis = out_axis
    shape = produced_blob_object.op_arg_blob_attr.shape
    op_conf.boxing_conf.split_box.part_num.extend(
        BalancedSplit(shape[out_axis], out_parallel_num)
    )
    op_conf.scope_symbol_id = oneflow.scope.current_scope().symbol_id
    return c_api_util.GetOpAttribute4OpConf(op_conf)


def GetConcatSplitBoxingParallelDescSymbol(
    builder, blob_parallel_desc_symbol, max_parallel_num
):
    random_rank_id = random.randint(0, max_parallel_num - 1)
    parallel_conf = placement_pb.ParallelConf()
    for machine_id, _ in blob_parallel_desc_symbol.machine_id2device_id_list.items():
        parallel_conf.device_name.append("%s:cpu:%s" % (machine_id, random_rank_id))
    return builder.GetParallelDescSymbol(parallel_conf)


def UnpackLogicalBoxingBlobObjectToPhysical(builder, produced_blob_object):
    if produced_blob_object.parallel_desc_symbol.parallel_num == 1:
        return [produced_blob_object]
    return builder.UnpackLogicalBlobToPhysicalBlobs(produced_blob_object)


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
    op_conf = op_conf_pb.OperatorConf()
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
    op_conf.scope_symbol_id = oneflow.scope.current_scope().symbol_id
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
    bn_in_op2blob_object["out"].InitOpArgBlobAttr(produced_blob_object.op_arg_blob_attr)
    return bn_in_op2blob_object["out"]


def _AssignOpConf():
    op_conf = op_conf_pb.OperatorConf()
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
    op_conf.scope_symbol_id = oneflow.scope.current_scope().symbol_id
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
    return boxing_middle.TryReplaceDeviceTag(builder, parallel_desc_symbol, device_tag)


def ReplaceDeviceTag(parallel_desc_symbol, device_tag, builder=None):
    return boxing_middle.ReplaceDeviceTag(
        parallel_desc_symbol, device_tag, builder=builder
    )


def _GetEagerNcclAllReduce(parallel_conf):
    op_conf = op_conf_pb.OperatorConf()
    op_conf.name = "eager_nccl_all_reduce"
    op_conf.user_conf.op_type_name = "eager_nccl_all_reduce"
    op_conf.user_conf.input["in"].s.append("eager_nccl_all_reduce/in_0")
    op_conf.user_conf.output["out"].s.append("eager_nccl_all_reduce/out_0")
    op_conf.user_conf.attr["parallel_conf"].at_string = str(parallel_conf)
    op_conf.scope_symbol_id = oneflow.scope.current_scope().symbol_id
    return c_api_util.GetOpAttribute4OpConf(op_conf)


def BalancedSplit(total, part_size):
    base = int(total / part_size)
    remainder = total % part_size
    return [base + int(i < remainder) for i in range(part_size)]
