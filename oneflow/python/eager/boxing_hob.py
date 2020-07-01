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
            x_blob_object, _ = ctx
            blob_object = BlobObject(
                object_id=x_blob_object.object_id,
                op_arg_parallel_attr=x_blob_object.op_arg_parallel_attr,
                op_arg_blob_attr=x_blob_object.op_arg_blob_attr,
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
            x_blob_object, op_arg_parallel_attr = ctx
            medium_blob_object = BlobObject(
                object_id=x_blob_object.object_id,
                op_arg_parallel_attr=self._GetMediumOpArgParallelAttr(ctx),
                op_arg_blob_attr=x_blob_object.op_arg_blob_attr,
                release=None,
            )
            value = (
                medium_blob_object,
                op_arg_parallel_attr,
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
    x_blob_object, op_arg_parallel_attr = context
    blob_device_ids = x_blob_object.parallel_desc_symbol.machine_id2device_id_list
    arg_parallel_desc_symbol = op_arg_parallel_attr.parallel_desc_symbol
    op_arg_device_ids = arg_parallel_desc_symbol.machine_id2device_id_list
    return list(blob_device_ids.keys()) == [0] and list(op_arg_device_ids.keys()) == [0]


@bool_functor("SameDeviceIds")
def SameDeviceIds(context):
    x_blob_object, op_arg_parallel_attr = context
    blob_device_ids = x_blob_object.parallel_desc_symbol.machine_id2device_id_list
    arg_parallel_desc_symbol = op_arg_parallel_attr.parallel_desc_symbol
    op_device_ids = arg_parallel_desc_symbol.machine_id2device_id_list
    return blob_device_ids == op_device_ids


@bool_functor("blob's devices contained in op_arg's devices")
def BlobDeviceIdsContainedInOpArgDevices(context):
    x_blob_object, op_arg_parallel_attr = context
    return op_arg_parallel_attr.parallel_desc_symbol.Containing(
        x_blob_object.parallel_desc_symbol
    )


@bool_functor("op_arg's devices contained in blob's devices")
def OpArgDeviceIdsContainedInBlobDevices(context):
    x_blob_object, op_arg_parallel_attr = context
    return x_blob_object.parallel_desc_symbol.Containing(
        op_arg_parallel_attr.parallel_desc_symbol
    )


@hob_context_attr("op_arg_sbp_parallel")
def op_arg_sbp_parallel(context):
    _, op_arg_parallel_attr = context
    return op_arg_parallel_attr.sbp_parallel


@hob_context_attr("blob_sbp_parallel")
def blob_sbp_parallel(context):
    x_blob_object, _ = context
    return x_blob_object.op_arg_parallel_attr.sbp_parallel


@hob_context_attr("blob_parallel_desc")
def blob_parallel_desc(context):
    x_blob_object, op_arg_parallel_attr = context
    return x_blob_object.op_arg_parallel_attr.parallel_desc_symbol


@hob_context_attr("op_arg_parallel_desc")
def op_arg_parallel_desc(context):
    x_blob_object, op_arg_parallel_attr = context
    return op_arg_parallel_attr.parallel_desc_symbol
