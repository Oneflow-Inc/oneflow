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

import oneflow.python.eager.symbol as symbol_util
import oneflow.core.job.sbp_parallel_pb2 as sbp_parallel_pb
import oneflow_api.oneflow.core.job.placement as placement_cfg
import oneflow_api.oneflow.core.common.shape as shape_proto_cfg
import oneflow_api
import random


class BoxingToMiddle(object):
    def __init__(
        self,
        boxing_method,
        get_middle_parallel_desc_symbol,
        get_middle_sbp_parallel,
        verbose=False,
    ):
        self.boxing_method_ = boxing_method
        self.get_middle_op_arg_parallel_attr_ = MiddleOpArgParallelAttr(
            get_middle_parallel_desc_symbol, get_middle_sbp_parallel,
        )
        self.verbose_ = verbose

    @property
    def boxing_method(self):
        return self.boxing_method_

    @property
    def get_middle_op_arg_parallel_attr(self):
        return self.get_middle_op_arg_parallel_attr_

    @property
    def verbose(self):
        return self.verbose_


def MiddleOpArgParallelAttr(get_parallel_desc_symbol, get_sbp_parallel):
    def GetOpArgParallelAttr(
        builder, produced_blob_object, consumer_op_arg_parallel_attr
    ):
        return oneflow_api.OpArgParallelAttribute(
            get_parallel_desc_symbol(
                builder, produced_blob_object, consumer_op_arg_parallel_attr
            ),
            str(
                get_sbp_parallel(
                    builder, produced_blob_object, consumer_op_arg_parallel_attr
                )
            ),
            str(produced_blob_object.op_arg_parallel_attr.opt_mirrored_parallel),
        )

    return GetOpArgParallelAttr


def ReplaceProducerDeviceTag(new_device_tag):
    def Getter(builder, produced_blob_object, consumer_op_arg_parallel_attr):
        x_parallel_attr = produced_blob_object.op_arg_parallel_attr
        return TryReplaceDeviceTag(
            builder, x_parallel_attr.parallel_desc_symbol, new_device_tag
        )

    return Getter


def ProducerRandomParallelIdPerMachine(device_tag=None):
    def Getter(builder, produced_blob_object, consumer_op_arg_parallel_attr):
        return RandomParallelIdPerMachine(
            produced_blob_object.parallel_desc_symbol,
            device_tag=device_tag,
            builder=builder,
        )

    return Getter


def ConsumerRandomParallelIdPerMachine(device_tag=None):
    def Getter(builder, produced_blob_object, consumer_op_arg_parallel_attr):
        return RandomParallelIdPerMachine(
            consumer_op_arg_parallel_attr.parallel_desc_symbol,
            device_tag=device_tag,
            builder=builder,
        )

    return Getter


def ProducerParallelDesc(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return produced_blob_object.parallel_desc_symbol


def ConsumerParallelDesc(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return consumer_op_arg_parallel_attr.parallel_desc_symbol


def ReplaceConsumerDeviceTag(new_device_tag):
    def Getter(builder, produced_blob_object, consumer_op_arg_parallel_attr):
        parallel_desc_sym = consumer_op_arg_parallel_attr.parallel_desc_symbol
        return TryReplaceDeviceTag(builder, parallel_desc_sym, new_device_tag)

    return Getter


def BroadcastParallel(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    sbp_parallel = sbp_parallel_pb.SbpParallel()
    sbp_parallel.broadcast_parallel.SetInParent()
    return sbp_parallel


def ProducerSbpParallel(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return produced_blob_object.op_arg_parallel_attr.sbp_parallel


def ConsumerSbpParallel(builder, produced_blob_object, consumer_op_arg_parallel_attr):
    return consumer_op_arg_parallel_attr.sbp_parallel


def TryReplaceDeviceTag(builder, parallel_desc_symbol, device_tag):
    if parallel_desc_symbol.device_tag == device_tag:
        return parallel_desc_symbol
    else:
        return ReplaceDeviceTag(parallel_desc_symbol, device_tag, builder=builder)


def ReplaceDeviceTag(parallel_desc_symbol, device_tag, builder=None):
    assert parallel_desc_symbol.device_tag != device_tag
    parallel_conf = placement_cfg.ParallelConf()
    parallel_conf.set_device_tag(device_tag)
    for device_name in parallel_desc_symbol.parallel_conf.device_name():
        parallel_conf.add_device_name(device_name)
    hierarchy = shape_proto_cfg.ShapeProto()
    for dim in parallel_desc_symbol.hierarchy:
        hierarchy.add_dim(dim)
    assert hierarchy.dim_size() > 0
    parallel_conf.mutable_hierarchy().CopyFrom(hierarchy)
    if builder is None:
        return oneflow_api.PlacementSymbol(
            parallel_desc_symbol.symbol_id, parallel_conf
        )
    else:
        return builder.GetParallelDescSymbol(parallel_conf)


def RandomParallelIdPerMachine(parallel_desc_symbol, device_tag=None, builder=None):
    if device_tag is None:
        device_tag = parallel_desc_symbol.parallel_conf.device_tag()
    assert device_tag is not None
    parallel_conf = placement_cfg.ParallelConf()
    parallel_conf.set_device_tag(device_tag)
    for machine_id, dev_ids in parallel_desc_symbol.machine_id2device_id_list.items():
        dev_id = dev_ids[random.randint(0, len(dev_ids) - 1)]
        parallel_conf.add_device_name("@%s:%s" % (machine_id, dev_id))
    if builder is None:
        return oneflow_api.PlacementSymbol(
            parallel_desc_symbol.symbol_id, parallel_conf
        )
    else:
        return builder.GetParallelDescSymbol(parallel_conf)
