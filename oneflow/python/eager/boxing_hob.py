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

from oneflow.python.lib.core.high_order_bool import bool_functor
from oneflow.python.lib.core.high_order_bool import hob_context_attr
from oneflow.python.lib.core.high_order_bool import BoolFunctor
import oneflow_api


class BoxingHobContext(object):
    def __init__(self, produced_blob_object, consumer_op_arg_parallel_attr):
        self.produced_blob_object_ = produced_blob_object
        self.consumer_op_arg_parallel_attr_ = consumer_op_arg_parallel_attr
        self.composer2lhs_context = {}
        self.composer2rhs_context = {}
        self.composer2middle_op_arg_parallel_attr = {}

    @property
    def produced_blob_object(self):
        return self.produced_blob_object_

    @property
    def consumer_op_arg_parallel_attr(self):
        return self.consumer_op_arg_parallel_attr_


class ComposeHob(BoolFunctor):
    def __init__(
        self, lhs_hob, rhs_hob, get_middle_op_arg_parallel_attr, middle_verbose_str=None
    ):
        self.get_middle_op_arg_parallel_attr_ = get_middle_op_arg_parallel_attr
        self.lhs_hob_ = lhs_hob
        self.rhs_hob_ = rhs_hob
        self.ctx_id2middle_op_arg_parallel_attr_ = {}
        self.middle_verbose_str_ = middle_verbose_str

    def verbose_debug_str(self, ctx, display_result=True):
        left_display = self.lhs_hob_.debug_str(self._GetLhsContext(ctx), display_result)
        display_result = display_result and self.lhs_hob_(self._GetLhsContext(ctx))
        right_display = self.rhs_hob_.debug_str(
            self._GetRhsContext(ctx), display_result
        )
        return "%s -> %s" % (left_display, right_display)

    def __call__(self, ctx):
        return self.lhs_hob_(self._GetLhsContext(ctx)) and self.rhs_hob_(
            self._GetRhsContext(ctx)
        )

    def _GetLhsContext(self, ctx):
        if self not in ctx.composer2lhs_context:
            blob_object = oneflow_api.BlobObject(
                ctx.produced_blob_object.object_id,
                ctx.produced_blob_object.op_arg_parallel_attr,
                ctx.produced_blob_object.op_arg_blob_attr,
            )
            value = BoxingHobContext(
                blob_object, self._GetMiddleOpArgParallelAttr(ctx),
            )
            ctx.composer2lhs_context[self] = value
        return ctx.composer2lhs_context[self]

    def _GetRhsContext(self, ctx):
        if self not in ctx.composer2rhs_context:
            middle_blob_object = oneflow_api.BlobObject(
                ctx.produced_blob_object.object_id,
                self._GetMiddleOpArgParallelAttr(ctx),
                ctx.produced_blob_object.op_arg_blob_attr,
            )
            value = BoxingHobContext(
                middle_blob_object, ctx.consumer_op_arg_parallel_attr,
            )
            ctx.composer2rhs_context[self] = value
        return ctx.composer2rhs_context[self]

    def _GetMiddleOpArgParallelAttr(self, ctx):
        if self not in ctx.composer2middle_op_arg_parallel_attr:
            value = self.get_middle_op_arg_parallel_attr_(
                None, ctx.produced_blob_object, ctx.consumer_op_arg_parallel_attr
            )
            if self.middle_verbose_str_ is not None:
                print("=== %s ===" % self.middle_verbose_str_)
                print(value)
            ctx.composer2middle_op_arg_parallel_attr[self] = value
        return ctx.composer2middle_op_arg_parallel_attr[self]


@bool_functor("SingleMachine")
def SingleMachine(ctx):
    blob_device_ids = dict(
        ctx.produced_blob_object.parallel_desc_symbol.machine_id2device_id_list
    )
    arg_parallel_desc_symbol = ctx.consumer_op_arg_parallel_attr.parallel_desc_symbol
    op_arg_device_ids = dict(arg_parallel_desc_symbol.machine_id2device_id_list)
    return list(blob_device_ids.keys()) == [0] and list(op_arg_device_ids.keys()) == [0]


@bool_functor("MatchDeviceOneToOnePerMachine")
def MatchDeviceOneToOnePerMachine(ctx):
    blob_device_ids = dict(
        ctx.produced_blob_object.parallel_desc_symbol.machine_id2device_id_list
    )
    arg_parallel_desc_symbol = ctx.consumer_op_arg_parallel_attr.parallel_desc_symbol
    op_arg_device_ids = dict(arg_parallel_desc_symbol.machine_id2device_id_list)
    if blob_device_ids.keys() != op_arg_device_ids.keys():
        return False
    for key in blob_device_ids.keys():
        if len(blob_device_ids[key]) != len(op_arg_device_ids[key]):
            return False
    return True


@bool_functor("Verbose")
def Verbose(ctx):
    print("============[producer]============")
    print(ctx.produced_blob_object.op_arg_parallel_attr.parallel_desc_symbol)
    print(ctx.produced_blob_object.op_arg_parallel_attr.sbp_parallel)
    print("============[consumer]============")
    print(ctx.consumer_op_arg_parallel_attr.parallel_desc_symbol)
    print(ctx.consumer_op_arg_parallel_attr.sbp_parallel)
    return True


@bool_functor("producer's devices contained in consumer's devices")
def ProducerDevicesContainedInConsumerDevices(ctx):
    return ctx.consumer_op_arg_parallel_attr.parallel_desc_symbol.Containing(
        ctx.produced_blob_object.parallel_desc_symbol
    )


@bool_functor("consumer's devices contained in producer's devices")
def ConsumerDevicesContainedInProducerDevices(ctx):
    return ctx.produced_blob_object.parallel_desc_symbol.Containing(
        ctx.consumer_op_arg_parallel_attr.parallel_desc_symbol
    )


@hob_context_attr("consumer_sbp_parallel")
def consumer_sbp_parallel(ctx):
    return ctx.consumer_op_arg_parallel_attr.sbp_parallel


@hob_context_attr("producer_sbp_parallel")
def producer_sbp_parallel(ctx):
    return ctx.produced_blob_object.op_arg_parallel_attr.sbp_parallel


@hob_context_attr("producer_parallel_desc")
def producer_parallel_desc(ctx):
    return ctx.produced_blob_object.op_arg_parallel_attr.parallel_desc_symbol


@hob_context_attr("consumer_parallel_desc")
def consumer_parallel_desc(ctx):
    return ctx.consumer_op_arg_parallel_attr.parallel_desc_symbol
