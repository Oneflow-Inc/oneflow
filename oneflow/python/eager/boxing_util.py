"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

from contextlib import contextmanager
import oneflow.python.eager.symbol as symbol_util
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.core.operator.op_attribute_pb2 as op_attribute_pb
import oneflow.core.job.sbp_parallel_pb2 as sbp_parallel_pb
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.balanced_splitter as balanced_splitter
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.lib.core.high_order_bool as high_order_bool
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.boxing_hob as boxing_hob
import oneflow.python.eager.op_infer_util as op_infer_util
from oneflow.python.eager.boxing_hob import BoxingHobContext
import oneflow.python.eager.boxing_middle as boxing_middle
import random
import oneflow
import oneflow_api.oneflow.core.job.placement as placement_cfg
import oneflow_api


def BoxingTo(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    hob_context = BoxingHobContext(produced_blob_object, consumer_op_arg_parallel_attr)
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

    global conditional_function_table
    function = enable_if.unique(
        conditional_function_table,
        context=BoxingHobContext(produced_blob_object, consumer_op_arg_parallel_attr),
        default=default,
    )
    return function(builder, produced_blob_object, consumer_op_arg_parallel_attr)


def boxing_condition(hob_expr, verbose=False):
    def Decorator(func):
        func.__oneflow_condition_hob__ = hob_expr
        if not verbose:
            hob_expr.__debug_str__ = GetBoxingDebugString(func)
        return func

    return Decorator


def FirstMatchedBoxing(*boxing_methods):
    hob_expr = enable_if.get_condition_hob(boxing_methods[0])
    for boxing_method in boxing_methods[1:]:
        hob_expr = hob_expr | enable_if.get_condition_hob(boxing_method)

    @enable_if.condition(hob_expr)
    def FirstMatched(builder, produced_blob_object, consumer_op_arg_parallel_attr):
        ctx = BoxingHobContext(produced_blob_object, consumer_op_arg_parallel_attr)
        for boxing_method in boxing_methods:
            hob_expr = enable_if.get_condition_hob(boxing_method)
            if not hob_expr(ctx):
                continue
            return boxing_method(
                builder, produced_blob_object, consumer_op_arg_parallel_attr
            )

    boxing_methods_names = [GetBoxingDebugString(m) for m in boxing_methods]
    FirstMatched.__debug_str__ = "(%s)" % (" | ".join(boxing_methods_names))
    return FirstMatched


def OptionalBoxing(boxing_method):
    opt_boxing_method = FirstMatchedBoxing(boxing_method, NoBoxing)
    debug_str = "Optional(%s)" % GetBoxingDebugString(boxing_method)
    opt_boxing_method.__debug_str__ = debug_str
    return opt_boxing_method


def ComposeBoxing(
    lhs_boxing, rhs_boxing, get_middle_op_arg_parallel_attr, middle_verbose_str=None
):
    composed_hob = boxing_hob.ComposeHob(
        enable_if.get_condition_hob(lhs_boxing),
        enable_if.get_condition_hob(rhs_boxing),
        get_middle_op_arg_parallel_attr=get_middle_op_arg_parallel_attr,
        middle_verbose_str=middle_verbose_str,
    )

    @enable_if.condition(composed_hob)
    def Composed(builder, produced_blob_object, consumer_op_arg_parallel_attr):
        tmp_op_arg_parallel_attr = get_middle_op_arg_parallel_attr(
            builder, produced_blob_object, consumer_op_arg_parallel_attr
        )
        tmp = lhs_boxing(builder, produced_blob_object, tmp_op_arg_parallel_attr)
        return rhs_boxing(builder, tmp, consumer_op_arg_parallel_attr)

    Composed.__debug_str__ = "%s->%s" % (
        GetBoxingDebugString(lhs_boxing),
        GetBoxingDebugString(rhs_boxing),
    )
    Composed.__left_debug_str__ = GetBoxingLeftDebugString(lhs_boxing)
    Composed.__right_debug_str__ = GetBoxingRightDebugString(rhs_boxing)
    return Composed


def GetBoxingDebugString(boxing_method):
    if hasattr(boxing_method, "__debug_str__"):
        return boxing_method.__debug_str__
    else:
        return boxing_method.__name__


def GetBoxingLeftDebugString(boxing_method):
    if hasattr(boxing_method, "__left_debug_str__"):
        return boxing_method.__left_debug_str__
    else:
        return GetBoxingDebugString(boxing_method)


def GetBoxingRightDebugString(boxing_method):
    if hasattr(boxing_method, "__right_debug_str__"):
        return boxing_method.__right_debug_str__
    else:
        return GetBoxingDebugString(boxing_method)


def Sequential(*boxing_methods, exclude=tuple(), middle_verbose=False):
    assert not isinstance(boxing_methods[-1], boxing_middle.BoxingToMiddle)
    composed = boxing_methods[-1]
    for boxing_to_middle in boxing_methods[-2::-1]:
        assert isinstance(boxing_to_middle, boxing_middle.BoxingToMiddle)
        if middle_verbose:
            middle_verbose_str = "middle op_arg_parallel_attr of %s->%s:" % (
                GetBoxingDebugString(boxing_to_middle.boxing_method),
                GetBoxingLeftDebugString(composed),
            )
        else:
            middle_verbose_str = None
        composed = ComposeBoxing(
            boxing_to_middle.boxing_method,
            composed,
            boxing_to_middle.get_middle_op_arg_parallel_attr,
            middle_verbose_str=middle_verbose_str,
        )
    if len(exclude) > 0:
        exclude_hob = enable_if.get_condition_hob(exclude[0])
        for method in exclude[1:]:
            exclude_hob = exclude_hob | enable_if.get_condition_hob(method)
        old_hob = enable_if.get_condition_hob(composed)
        enable_if.set_condition_hob(composed, old_hob & ~exclude_hob)
    return composed


MatchCopyH2D = (
    (
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


@boxing_condition(MatchCopyH2D)
def CopyH2D(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return CopyHD(builder, produced_blob_object, consumer_op_arg_parallel_attr)


MatchCopyD2H = (
    (
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


@boxing_condition(MatchCopyD2H)
def CopyD2H(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return CopyHD(builder, produced_blob_object, consumer_op_arg_parallel_attr)


def CopyHD(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    arg_parallel_desc_symbol = consumer_op_arg_parallel_attr.parallel_desc_symbol
    op_device_tag = arg_parallel_desc_symbol.device_tag
    return BuildCopyHdInstruction(builder, produced_blob_object, op_device_tag)


BlobIsPartialSum = boxing_hob.producer_sbp_parallel.HasField("partial_sum_parallel")
OpArgIsBroadcast = boxing_hob.consumer_sbp_parallel.HasField("broadcast_parallel")


MatchInterNodeOneToMany = (
    ~boxing_hob.SingleMachine
    & (boxing_hob.producer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.consumer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.producer_parallel_desc.parallel_num == 1)
    & (boxing_hob.consumer_parallel_desc.parallel_num > 1)
    & OpArgIsBroadcast
)


@boxing_condition(MatchInterNodeOneToMany)
def InterNodeOneToMany(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    out_blobs = []
    consumer_dev_ids = (
        consumer_op_arg_parallel_attr.parallel_desc_symbol.machine_id2device_id_list
    )
    for machine_id, device_ids in consumer_dev_ids.items():
        for device_id in device_ids:
            parallel_conf = placement_cfg.ParallelConf()
            parallel_conf.set_device_tag("cpu")
            parallel_conf.add_device_name("%s:%s" % (machine_id, device_id))
            parallel_desc_symbol = builder.GetParallelDescSymbol(parallel_conf)
            out_blob = builder.Build121To(produced_blob_object, parallel_desc_symbol)
            out_blobs.append(out_blob)

    return PackPhysicalBoxingBlobObjectsToLogical(
        builder,
        out_blobs,
        consumer_op_arg_parallel_attr,
        produced_blob_object.op_arg_blob_attr,
    )


MatchInterNodeOneToOne = (
    (boxing_hob.producer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.consumer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.producer_parallel_desc != boxing_hob.consumer_parallel_desc)
    & (
        boxing_hob.producer_parallel_desc.parallel_num
        == boxing_hob.consumer_parallel_desc.parallel_num
    )
    & ~boxing_hob.MatchDeviceOneToOnePerMachine
    & (
        (boxing_hob.producer_sbp_parallel == boxing_hob.consumer_sbp_parallel)
        | (boxing_hob.producer_parallel_desc.parallel_num == 1)
    )
)


@boxing_condition(MatchInterNodeOneToOne)
def InterNodeOneToOne(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return builder.Build121To(
        produced_blob_object, consumer_op_arg_parallel_attr.parallel_desc_symbol
    )


MatchCpuBroadcastOneToOne = (
    (boxing_hob.producer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.consumer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.producer_parallel_desc != boxing_hob.consumer_parallel_desc)
    & boxing_hob.MatchDeviceOneToOnePerMachine
    & (
        (boxing_hob.producer_sbp_parallel == boxing_hob.consumer_sbp_parallel)
        | (boxing_hob.producer_parallel_desc.parallel_num == 1)
    )
)


@boxing_condition(MatchCpuBroadcastOneToOne)
def CpuBroadcastOneToOne(builder, produced_blob_object, consumer_op_arg_parallel_attr):
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
    boxing_hob.producer_parallel_desc == boxing_hob.consumer_parallel_desc
) & (
    (boxing_hob.producer_sbp_parallel == boxing_hob.consumer_sbp_parallel)
    | (boxing_hob.producer_parallel_desc.parallel_num == 1)
)


@boxing_condition(MatchNoBoxing)
def NoBoxing(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return produced_blob_object


@boxing_condition(boxing_hob.Verbose & MatchNoBoxing)
def VerboseNoBoxing(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return produced_blob_object


def VerboseOptionalBoxing(boxing_method):
    opt_boxing_method = FirstMatchedBoxing(boxing_method, VerboseNoBoxing)
    debug_str = "VerboseOptional(%s)" % GetBoxingDebugString(boxing_method)
    opt_boxing_method.__debug_str__ = debug_str
    return opt_boxing_method


MatchNcclAllReduce = (
    boxing_hob.SingleMachine
    & (boxing_hob.producer_parallel_desc.device_tag == "gpu")
    & (boxing_hob.producer_parallel_desc == boxing_hob.consumer_parallel_desc)
    & (boxing_hob.consumer_parallel_desc.parallel_num > 1)
    & BlobIsPartialSum
    & OpArgIsBroadcast
)


@boxing_condition(MatchNcclAllReduce)
def GpuNcclAllReduce(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    parallel_conf = consumer_op_arg_parallel_attr.parallel_desc_symbol.parallel_conf
    bn_in_op2blob_object = oneflow_api.deprecated.BnInOp2BlobObject()
    bn_in_op2blob_object["in_0"] = produced_blob_object
    op_attribute = _GetEagerNcclAllReduce(parallel_conf, bn_in_op2blob_object)
    cfg_op_attribute = oneflow_api.deprecated.MakeOpAttributeByString(str(op_attribute))
    builder.NoBoxingStatelessCall(
        cfg_op_attribute,
        parallel_conf,
        bn_in_op2blob_object,
        blob_cache_util.FindOrCreateDelegateBlobObject,
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


MatchNaiveCpuSplitToSplit = (
    (boxing_hob.producer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.consumer_parallel_desc.device_tag == "cpu")
    & (MatchSplitOneToMany | MatchConcatManyToOne | MatchConcatManyToSplitMany)
)


@boxing_condition(MatchNaiveCpuSplitToSplit)
def NaiveCpuSplitToSplit(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return NaiveCpuRefPhysicalBlobObjectsScope(
        builder,
        produced_blob_object,
        consumer_op_arg_parallel_attr,
        get_physical_out_blob_objects=NaiveBoxingToPhysicalBlobObjects,
    )


MatchNaiveCpuPartialSumToSplit = (
    (boxing_hob.producer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.consumer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.producer_parallel_desc.parallel_num > 1)
    & boxing_hob.producer_sbp_parallel.HasField("partial_sum_parallel")
    & (
        (boxing_hob.consumer_parallel_desc.parallel_num == 1)
        | boxing_hob.consumer_sbp_parallel.HasField("split_parallel")
    )
)


@boxing_condition(MatchNaiveCpuPartialSumToSplit)
def NaiveCpuPartialSumToSplit(
    builder, produced_blob_object, consumer_op_arg_parallel_attr
):
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
    consumer_parallel_desc_symbol = consumer_op_arg_parallel_attr.parallel_desc_symbol
    out_parallel_num = consumer_parallel_desc_symbol.parallel_num
    boxing_parallel_desc_symbol = GetConcatSplitBoxingParallelDescSymbol(
        builder,
        consumer_parallel_desc_symbol,
        max(len(physical_in_blob_objects), out_parallel_num),
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
    op_attribute = ConstructNaiveBoxingOpConf(
        produced_blob_object,
        consumer_op_arg_parallel_attr,
        len(physical_in_blob_objects),
        out_parallel_num,
    )
    return BuildNaiveCpuBoxing(
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


def BuildNaiveCpuBoxing(
    builder,
    op_attribute,
    physical_in_blob_objects,
    boxing_parallel_desc_symbol,
    out_parallel_num,
):
    bn_in_op2blob_object = oneflow_api.deprecated.BnInOp2BlobObject()
    for i in range(len(physical_in_blob_objects)):
        bn_in_op2blob_object["in_%s" % i] = physical_in_blob_objects[i]
    cfg_op_attribute = oneflow_api.deprecated.MakeOpAttributeByString(str(op_attribute))
    builder.NoBoxingStatelessCall(
        cfg_op_attribute,
        boxing_parallel_desc_symbol.parallel_conf,
        bn_in_op2blob_object,
        blob_cache_util.FindOrCreateDelegateBlobObject,
    )
    return [bn_in_op2blob_object["out_%s" % i] for i in range(out_parallel_num)]


# S -> S or P -> S
def ConstructNaiveBoxingOpConf(
    produced_blob_object,
    consumer_op_arg_parallel_attr,
    in_parallel_num,
    out_parallel_num,
):
    op_conf = op_conf_pb.OperatorConf()
    op_conf.name = "undefined_boxing_op_name"
    op_conf.device_tag = "cpu"
    op_conf.boxing_conf.lbi.op_name = "undefined_boxing_op_name"
    op_conf.boxing_conf.lbi.blob_name = "undefined_boxing_blob_name"
    op_conf.boxing_conf.in_num = in_parallel_num
    op_conf.boxing_conf.out_num = out_parallel_num
    in_sbp_parallel = produced_blob_object.op_arg_parallel_attr.sbp_parallel
    if in_sbp_parallel.has_split_parallel():
        op_conf.boxing_conf.concat_box.axis = in_sbp_parallel.split_parallel().axis()
    elif in_parallel_num == 1:
        op_conf.boxing_conf.concat_box.axis = 0
    else:
        assert in_sbp_parallel.has_partial_sum_parallel()
        op_conf.boxing_conf.add_box.SetInParent()
    out_sbp_parallel = consumer_op_arg_parallel_attr.sbp_parallel
    if out_sbp_parallel.has_split_parallel():
        out_axis = out_sbp_parallel.split_parallel().axis()
    else:
        assert out_parallel_num == 1
        out_axis = 0
    op_conf.boxing_conf.split_box.axis = out_axis
    shape = produced_blob_object.op_arg_blob_attr.shape
    op_conf.boxing_conf.split_box.part_num.extend(
        balanced_splitter.BalancedPartNums(shape[out_axis], out_parallel_num)
    )
    bn_in_op2blob_object = oneflow_api.deprecated.BnInOp2BlobObject()
    for i in range(in_parallel_num):
        bn_in_op2blob_object["in_%s" % i] = produced_blob_object
    return op_infer_util.Infer(op_conf, bn_in_op2blob_object)


def GetConcatSplitBoxingParallelDescSymbol(
    builder, blob_parallel_desc_symbol, max_parallel_num
):
    random_rank_id = random.randint(0, max_parallel_num - 1)
    parallel_conf = placement_cfg.ParallelConf()
    parallel_conf.set_device_tag("cpu")
    for machine_id, _ in blob_parallel_desc_symbol.machine_id2device_id_list.items():
        parallel_conf.add_device_name("%s:%s" % (machine_id, random_rank_id))
    return builder.GetParallelDescSymbol(parallel_conf)


def UnpackLogicalBoxingBlobObjectToPhysical(builder, produced_blob_object):
    if produced_blob_object.parallel_desc_symbol.parallel_num == 1:
        return [produced_blob_object]
    return builder.UnpackLogicalBlobToPhysicalBlobs(produced_blob_object)


MatchCpuBroadcastOneToMany = (
    boxing_hob.SingleMachine
    & (boxing_hob.producer_parallel_desc.device_tag == "cpu")
    & (boxing_hob.consumer_parallel_desc.device_tag == "cpu")
    & boxing_hob.ProducerDevicesContainedInConsumerDevices
    & (boxing_hob.producer_parallel_desc.parallel_num == 1)
    & (boxing_hob.consumer_parallel_desc.parallel_num > 1)
    & boxing_hob.consumer_sbp_parallel.HasField("broadcast_parallel")
)


@boxing_condition(MatchCpuBroadcastOneToMany)
def CpuBroadcastOneToMany(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return CpuOneToManyBroadcastBlobReference(
        builder,
        produced_blob_object,
        consumer_op_arg_parallel_attr.parallel_desc_symbol,
    )


MatchBroadcastManyToOne = (
    (
        boxing_hob.producer_parallel_desc.device_tag
        == boxing_hob.consumer_parallel_desc.device_tag
    )
    & boxing_hob.ConsumerDevicesContainedInProducerDevices
    & (boxing_hob.producer_parallel_desc.parallel_num > 1)
    & (boxing_hob.consumer_parallel_desc.parallel_num == 1)
    & boxing_hob.producer_sbp_parallel.HasField("broadcast_parallel")
)


@boxing_condition(MatchBroadcastManyToOne)
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


def CpuOneToManyBroadcastBlobReference(
    builder, produced_blob_object, to_parallel_desc_symbol
):
    x_parallel_desc_symbol = produced_blob_object.parallel_desc_symbol
    x_machine_ids = list(dict(x_parallel_desc_symbol.machine_id2device_id_list).keys())
    to_machine_ids = list(
        dict(to_parallel_desc_symbol.machine_id2device_id_list).keys()
    )
    assert x_machine_ids == to_machine_ids, (x_machine_ids, to_machine_ids)
    x_first_device_ids = x_parallel_desc_symbol.machine_id2device_id_list[
        x_machine_ids[0]
    ]
    assert len(x_first_device_ids) == 1, x_first_device_ids
    if x_parallel_desc_symbol == to_parallel_desc_symbol:
        return produced_blob_object
    return builder.BroadcastBlobReference(produced_blob_object, to_parallel_desc_symbol)


def BuildCopyHdInstruction(builder, produced_blob_object, to_device_tag):
    op_conf, lbi = _MakeCopyHdOpConfAndRetLbi()
    return _BuildCopyInstruction(builder, produced_blob_object, op_conf, to_device_tag)


def _MakeCopyHdOpConfAndRetLbi():
    op_conf = op_conf_pb.OperatorConf()
    op_conf.name = "copy_hd"
    op_conf.device_tag = "gpu"
    setattr(op_conf.copy_conf, "in", "%s/in" % op_conf.name)
    op_conf.copy_conf.out = "out"
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return op_conf, lbi


@contextmanager
def _CudaHostPinBlob(build, blob_object):
    build.CudaHostRegisterBlob(blob_object)
    try:
        yield
    finally:
        build.CudaHostUnregisterBlob(blob_object)


def _BuildCopyInstruction(builder, produced_blob_object, op_conf, to_device_tag):
    x_devices = produced_blob_object.parallel_desc_symbol.machine_id2device_id_list
    x_device_tag = produced_blob_object.parallel_desc_symbol.device_tag
    bn_in_op2blob_object = oneflow_api.deprecated.BnInOp2BlobObject()
    bn_in_op2blob_object["in"] = produced_blob_object
    op_attribute = op_infer_util.Infer(op_conf, bn_in_op2blob_object)
    assert to_device_tag != x_device_tag, (to_device_tag, x_device_tag)
    cfg_op_attribute = oneflow_api.deprecated.MakeOpAttributeByString(str(op_attribute))
    if to_device_tag == "cpu" and x_device_tag == "gpu":
        x_parallel_conf = produced_blob_object.parallel_desc_symbol.parallel_conf
        builder.NoBoxingCudaD2HStatelessCall(
            cfg_op_attribute, x_parallel_conf, bn_in_op2blob_object, TryReplaceDeviceTag
        )
    elif to_device_tag == "gpu" and x_device_tag == "cpu":
        out_parallel_desc_symbol = TryReplaceDeviceTag(
            builder, produced_blob_object.parallel_desc_symbol, to_device_tag
        )
        out_parallel_conf = out_parallel_desc_symbol.parallel_conf
        with _CudaHostPinBlob(builder, produced_blob_object):
            builder.NoBoxingCudaH2DStatelessCall(
                cfg_op_attribute, out_parallel_conf, bn_in_op2blob_object,
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
    op_conf = op_conf_pb.OperatorConf()
    op_conf.name = "assign"
    op_conf.assign_conf.ref = "assign/ref"
    op_conf.assign_conf.value = "assign/value"
    device_tag = oneflow.current_scope().device_parallel_desc_symbol.device_tag
    op_conf.device_tag = device_tag
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
    bn_in_op2blob_object = oneflow_api.deprecated.BnInOp2BlobObject()
    bn_in_op2blob_object["ref"] = ref_blob_object
    bn_in_op2blob_object["value"] = value_blob_object
    op_attribute = op_infer_util.Infer(op_conf, bn_in_op2blob_object)
    cfg_op_attribute = oneflow_api.deprecated.MakeOpAttributeByString(str(op_attribute))
    if ref_device_tag == value_device_tag:
        builder.NoBoxingStatelessCall(
            cfg_op_attribute,
            ref_parallel_conf,
            bn_in_op2blob_object,
            blob_cache_util.FindOrCreateDelegateBlobObject,
        )
    elif ref_device_tag == "cpu" and value_device_tag == "gpu":
        value_parallel_conf = value_blob_object.parallel_desc_symbol.parallel_conf
        builder.NoBoxingCudaD2HStatelessCall(
            cfg_op_attribute,
            value_parallel_conf,
            bn_in_op2blob_object,
            TryReplaceDeviceTag,
        )
    elif ref_device_tag == "gpu" and value_device_tag == "cpu":
        with _CudaHostPinBlob(builder, value_blob_object):
            builder.NoBoxingCudaH2DStatelessCall(
                cfg_op_attribute, ref_parallel_conf, bn_in_op2blob_object,
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


def _GetEagerNcclAllReduce(parallel_conf, ibn2blob_object):
    op_conf = op_conf_pb.OperatorConf()
    op_conf.device_tag = "gpu"
    op_conf.name = "eager_nccl_all_reduce"
    op_conf.user_conf.op_type_name = "eager_nccl_all_reduce"
    op_conf.user_conf.input["in"].s.append("eager_nccl_all_reduce/in_0")
    op_conf.user_conf.output["out"].s.append("eager_nccl_all_reduce/out_0")
    op_conf.user_conf.attr["parallel_conf"].at_string = str(parallel_conf)
    return op_infer_util.Infer(op_conf, ibn2blob_object)


NcclAllReduce = Sequential(
    boxing_middle.BoxingToMiddle(
        GpuNcclAllReduce,
        boxing_middle.ProducerParallelDesc,
        boxing_middle.BroadcastParallel,
    ),
    OptionalBoxing(CopyD2H),
)

BoxingIntraNodeOneToOne = Sequential(
    boxing_middle.BoxingToMiddle(
        OptionalBoxing(CopyD2H),
        boxing_middle.ReplaceProducerDeviceTag("cpu"),
        boxing_middle.ProducerSbpParallel,
    ),
    boxing_middle.BoxingToMiddle(
        CpuBroadcastOneToOne,
        boxing_middle.ReplaceConsumerDeviceTag("cpu"),
        boxing_middle.ConsumerSbpParallel,
    ),
    OptionalBoxing(CopyH2D),
)

BoxingInterNodeOneToOne = Sequential(
    boxing_middle.BoxingToMiddle(
        OptionalBoxing(CopyD2H),
        boxing_middle.ReplaceProducerDeviceTag("cpu"),
        boxing_middle.ProducerSbpParallel,
    ),
    boxing_middle.BoxingToMiddle(
        InterNodeOneToOne,
        boxing_middle.ReplaceConsumerDeviceTag("cpu"),
        boxing_middle.ConsumerSbpParallel,
    ),
    OptionalBoxing(CopyH2D),
)

BoxingInterNodeOneToMany = Sequential(
    boxing_middle.BoxingToMiddle(
        OptionalBoxing(CopyD2H),
        boxing_middle.ReplaceProducerDeviceTag("cpu"),
        boxing_middle.ProducerSbpParallel,
    ),
    boxing_middle.BoxingToMiddle(
        InterNodeOneToMany,
        boxing_middle.ReplaceConsumerDeviceTag("cpu"),
        boxing_middle.ConsumerSbpParallel,
    ),
    OptionalBoxing(CopyH2D),
)

conditional_function_table = [
    CopyH2D,
    CopyD2H,
    NoBoxing,
    # one to one
    BoxingIntraNodeOneToOne,
    BoxingInterNodeOneToOne,
    BoxingInterNodeOneToMany,
    # B -> B
    BroadcastManyToOne,
    Sequential(
        boxing_middle.BoxingToMiddle(
            OptionalBoxing(BroadcastManyToOne),
            boxing_middle.ProducerRandomParallelIdPerMachine(),
            boxing_middle.ProducerSbpParallel,
        ),
        boxing_middle.BoxingToMiddle(
            OptionalBoxing(CopyD2H),
            boxing_middle.ReplaceProducerDeviceTag("cpu"),
            boxing_middle.ProducerSbpParallel,
        ),
        boxing_middle.BoxingToMiddle(
            OptionalBoxing(CpuBroadcastOneToOne),
            boxing_middle.ConsumerRandomParallelIdPerMachine("cpu"),
            boxing_middle.BroadcastParallel,
        ),
        boxing_middle.BoxingToMiddle(
            OptionalBoxing(CpuBroadcastOneToMany),
            boxing_middle.ReplaceConsumerDeviceTag("cpu"),
            boxing_middle.BroadcastParallel,
        ),
        OptionalBoxing(CopyH2D),
        exclude=(
            BroadcastManyToOne,
            CopyH2D,
            CopyD2H,
            NoBoxing,
            BoxingIntraNodeOneToOne,
        ),
    ),
    # B -> S
    Sequential(
        boxing_middle.BoxingToMiddle(
            BroadcastManyToOne,
            boxing_middle.ProducerRandomParallelIdPerMachine(),
            boxing_middle.ProducerSbpParallel,
        ),
        boxing_middle.BoxingToMiddle(
            OptionalBoxing(CopyD2H),
            boxing_middle.ReplaceProducerDeviceTag("cpu"),
            boxing_middle.ProducerSbpParallel,
        ),
        boxing_middle.BoxingToMiddle(
            NaiveCpuSplitToSplit,
            boxing_middle.ReplaceConsumerDeviceTag("cpu"),
            boxing_middle.ConsumerSbpParallel,
        ),
        OptionalBoxing(CopyH2D),
    ),
    # P -> B
    NcclAllReduce,  # e.g. gpu, 0:0-3 -> gpu, 0:0-3
    Sequential(
        boxing_middle.BoxingToMiddle(
            OptionalBoxing(CopyD2H),
            boxing_middle.ReplaceProducerDeviceTag("cpu"),
            boxing_middle.ProducerSbpParallel,
        ),
        boxing_middle.BoxingToMiddle(
            NaiveCpuPartialSumToSplit,
            boxing_middle.ConsumerRandomParallelIdPerMachine("cpu"),
            boxing_middle.BroadcastParallel,
        ),
        boxing_middle.BoxingToMiddle(
            CpuBroadcastOneToMany,
            boxing_middle.ReplaceConsumerDeviceTag("cpu"),
            boxing_middle.BroadcastParallel,
        ),
        OptionalBoxing(CopyH2D),
        exclude=(NcclAllReduce,),
    ),
    # P -> S
    Sequential(
        boxing_middle.BoxingToMiddle(
            OptionalBoxing(CopyD2H),
            boxing_middle.ReplaceProducerDeviceTag("cpu"),
            boxing_middle.ProducerSbpParallel,
        ),
        boxing_middle.BoxingToMiddle(
            NaiveCpuPartialSumToSplit,
            boxing_middle.ReplaceConsumerDeviceTag("cpu"),
            boxing_middle.ConsumerSbpParallel,
        ),
        OptionalBoxing(CopyH2D),
    ),
    # S -> B
    Sequential(
        boxing_middle.BoxingToMiddle(
            OptionalBoxing(CopyD2H),
            boxing_middle.ReplaceProducerDeviceTag("cpu"),
            boxing_middle.ProducerSbpParallel,
        ),
        boxing_middle.BoxingToMiddle(
            NaiveCpuSplitToSplit,
            boxing_middle.ConsumerRandomParallelIdPerMachine("cpu"),
            boxing_middle.BroadcastParallel,
        ),
        boxing_middle.BoxingToMiddle(
            CpuBroadcastOneToMany,
            boxing_middle.ReplaceConsumerDeviceTag("cpu"),
            boxing_middle.BroadcastParallel,
        ),
        OptionalBoxing(CopyH2D),
        exclude=(NcclAllReduce,),
    ),
    # S -> S
    Sequential(
        boxing_middle.BoxingToMiddle(
            OptionalBoxing(CopyD2H),
            boxing_middle.ReplaceProducerDeviceTag("cpu"),
            boxing_middle.ProducerSbpParallel,
        ),
        boxing_middle.BoxingToMiddle(
            NaiveCpuSplitToSplit,
            boxing_middle.ReplaceConsumerDeviceTag("cpu"),
            boxing_middle.ConsumerSbpParallel,
        ),
        OptionalBoxing(CopyH2D),
    ),
]
