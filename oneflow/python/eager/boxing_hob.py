from __future__ import absolute_import

from oneflow.python.lib.core.high_order_bool import bool_functor
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


@bool_functor("SameSbpParallel")
def SameSbpParallel(context):
    x_blob_object, op_arg_parallel_attr = context
    x_sbp_parallel = x_blob_object.op_arg_parallel_attr.sbp_parallel
    op_arg_sbp_parallel = op_arg_parallel_attr.sbp_parallel
    return x_sbp_parallel == op_arg_sbp_parallel


@bool_functor("HostToGpuDevice")
def HostToGpuDevice(context):
    x_blob_object, op_arg_parallel_attr = context
    blob_device_tag = x_blob_object.parallel_desc_symbol.device_tag
    op_device_tag = op_arg_parallel_attr.parallel_desc_symbol.device_tag
    return blob_device_tag == "cpu" and op_device_tag == "gpu"


def BlobOnDevice(device_tag):
    @bool_functor("blob on device %s" % device_tag)
    def BoolFunctor(context):
        x_blob_object, op_arg_parallel_attr = context
        return x_blob_object.parallel_desc_symbol.device_tag == device_tag

    return BoolFunctor


def OpArgOnDevice(device_tag):
    @bool_functor("operator arg on device %s" % device_tag)
    def BoolFunctor(context):
        x_blob_object, op_arg_parallel_attr = context
        return op_arg_parallel_attr.parallel_desc_symbol.device_tag == device_tag

    return BoolFunctor


@bool_functor("GpuDeviceToHost")
def GpuDeviceToHost(context):
    x_blob_object, op_arg_parallel_attr = context
    blob_device_tag = x_blob_object.parallel_desc_symbol.device_tag
    op_device_tag = op_arg_parallel_attr.parallel_desc_symbol.device_tag
    return blob_device_tag == "gpu" and op_device_tag == "cpu"


@bool_functor("SameParallelDesc")
def SameParallelDesc(context):
    x_blob_object, op_arg_parallel_attr = context
    x_parallel_desc_sym = x_blob_object.parallel_desc_symbol
    op_arg_parallel_desc_sym = op_arg_parallel_attr.parallel_desc_symbol
    return x_parallel_desc_sym == op_arg_parallel_desc_sym


@bool_functor("BlobOnSingleDevice")
def BlobOnSingleDevice(context):
    x_blob_object, op_arg_parallel_attr = context
    return x_blob_object.parallel_desc_symbol.parallel_num == 1


@bool_functor("SameMachineId")
def SameMachineId(context):
    x_blob_object, op_arg_parallel_attr = context
    x_parallel_desc_sym = x_blob_object.parallel_desc_symbol
    return (
        x_parallel_desc_sym.machine_id2device_id_list.keys()
        == x_parallel_desc_sym.machine_id2device_id_list.keys()
    )


@bool_functor("BlobPartialSumParallel")
def BlobPartialSumParallel(context):
    x_blob_object, _ = context
    sbp_parallel = x_blob_object.op_arg_parallel_attr.sbp_parallel
    return sbp_parallel.HasField("partial_sum_parallel")


@bool_functor("OpArgParallelNumGT1")
def OpArgParallelNumGT1(context):
    x_blob_object, op_arg_parallel_attr = context
    op_arg_parallel_desc_sym = op_arg_parallel_attr.parallel_desc_symbol
    return op_arg_parallel_desc_sym.parallel_num > 1


@bool_functor("OpArgBroadcastParallel")
def OpArgBroadcastParallel(context):
    x_blob_object, op_arg_parallel_attr = context
    return op_arg_parallel_attr.sbp_parallel.HasField("broadcast_parallel")
