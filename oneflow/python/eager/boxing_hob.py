from __future__ import absolute_import

from oneflow.python.lib.core.high_order_bool import bool_functor
from oneflow.python.lib.core.high_order_bool import hob_context_attr
from oneflow.python.lib.core.high_order_bool import BoolFunctor
from oneflow.python.eager.object import BlobObject


class ComposeHob(BoolFunctor):
    def __init__(self, lhs_hob, rhs_hob, get_medium_op_arg_parallel_attr):
        self.get_medium_op_arg_parallel_attr_ = get_medium_op_arg_parallel_attr
        self.lhs_hob_ = lhs_hob
        self.rhs_hob_ = rhs_hob
        self.ctx_id2medium_op_arg_parallel_attr_ = {}
        self.ctx_id2lhs_context_ = {}
        self.ctx_id2rhs_context_ = {}

    def debug_str(self, ctx, display_result=True):
        left_display = self.lhs_hob_.debug_str(self._GetLhsContext(ctx), display_result)
        display_result = display_result and self.lhs_hob_(self._GetLhsContext(ctx))
        right_display = self.rhs_hob_.debug_str(
            self._GetRhsContext(ctx), display_result
        )
        medium_name = self.get_medium_op_arg_parallel_attr_.__name__
        if hasattr(self.get_medium_op_arg_parallel_attr_, "__debug_str__"):
            medium_name = self.get_medium_op_arg_parallel_attr_.__debug_str__
        return "{%s} (->%s->) {%s}" % (left_display, medium_name, right_display)

    def __call__(self, ctx):
        return self.lhs_hob_(self._GetLhsContext(ctx)) and self.rhs_hob_(
            self._GetRhsContext(ctx)
        )

    def _GetLhsContext(self, ctx):
        ctx_id = tuple(map(id, ctx))
        if ctx_id not in self.ctx_id2lhs_context_:
            produced_blob_object, _ = ctx
            blob_object = BlobObject(
                object_id=produced_blob_object.object_id,
                op_arg_parallel_attr=produced_blob_object.op_arg_parallel_attr,
                op_arg_blob_attr=produced_blob_object.op_arg_blob_attr,
                release=None,
            )
            value = (
                blob_object,
                self._GetMediumOpArgParallelAttr(ctx),
            )
            self.ctx_id2lhs_context_[ctx_id] = value
        return self.ctx_id2lhs_context_[ctx_id]

    def _GetRhsContext(self, ctx):
        ctx_id = tuple(map(id, ctx))
        if ctx_id not in self.ctx_id2rhs_context_:
            produced_blob_object, consumer_op_arg_parallel_attr = ctx
            medium_blob_object = BlobObject(
                object_id=produced_blob_object.object_id,
                op_arg_parallel_attr=self._GetMediumOpArgParallelAttr(ctx),
                op_arg_blob_attr=produced_blob_object.op_arg_blob_attr,
                release=None,
            )
            value = (
                medium_blob_object,
                consumer_op_arg_parallel_attr,
            )
            self.ctx_id2rhs_context_[ctx_id] = value
        return self.ctx_id2rhs_context_[ctx_id]

    def _GetMediumOpArgParallelAttr(self, ctx):
        ctx_id = tuple(map(id, ctx))
        if ctx_id not in self.ctx_id2medium_op_arg_parallel_attr_:
            value = self.get_medium_op_arg_parallel_attr_(None, *ctx)
            self.ctx_id2medium_op_arg_parallel_attr_[ctx_id] = value
        return self.ctx_id2medium_op_arg_parallel_attr_[ctx_id]


@bool_functor("MasterMachineOnly")
def MasterMachineOnly(context):
    produced_blob_object, consumer_op_arg_parallel_attr = context
    blob_device_ids = (
        produced_blob_object.parallel_desc_symbol.machine_id2device_id_list
    )
    arg_parallel_desc_symbol = consumer_op_arg_parallel_attr.parallel_desc_symbol
    op_arg_device_ids = arg_parallel_desc_symbol.machine_id2device_id_list
    return list(blob_device_ids.keys()) == [0] and list(op_arg_device_ids.keys()) == [0]


@bool_functor("producer's devices contained in consumer's devices")
def ProducerDevicesContainedInConsumerDevices(context):
    produced_blob_object, consumer_op_arg_parallel_attr = context
    return consumer_op_arg_parallel_attr.parallel_desc_symbol.Containing(
        produced_blob_object.parallel_desc_symbol
    )


@bool_functor("consumer's devices contained in producer's devices")
def ConsumerDevicesContainedInProducerDevices(context):
    produced_blob_object, consumer_op_arg_parallel_attr = context
    return produced_blob_object.parallel_desc_symbol.Containing(
        consumer_op_arg_parallel_attr.parallel_desc_symbol
    )


@hob_context_attr("consumer_sbp_parallel")
def consumer_sbp_parallel(context):
    _, consumer_op_arg_parallel_attr = context
    return consumer_op_arg_parallel_attr.sbp_parallel


@hob_context_attr("producer_sbp_parallel")
def producer_sbp_parallel(context):
    produced_blob_object, _ = context
    return produced_blob_object.op_arg_parallel_attr.sbp_parallel


@hob_context_attr("producer_parallel_desc")
def producer_parallel_desc(context):
    produced_blob_object, consumer_op_arg_parallel_attr = context
    return produced_blob_object.op_arg_parallel_attr.parallel_desc_symbol


@hob_context_attr("consumer_parallel_desc")
def consumer_parallel_desc(context):
    produced_blob_object, consumer_op_arg_parallel_attr = context
    return consumer_op_arg_parallel_attr.parallel_desc_symbol
