from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_ctx
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.blob_register as blob_register_util
import oneflow

blob_register = blob_register_util.GetDefaultBlobRegister()


def Interpret(op_conf, scope_symbol=None):
    if scope_symbol is None:
        scope_symbol = oneflow.scope.current_scope()
    func = enable_if.unique([LazyInfer, EagerInferAndRun])
    return func(op_conf, scope_symbol)


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def LazyInfer(op_conf, scope_symbol=None):
    return compile_ctx.CurJobAddOp(op_conf, scope_symbol)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def EagerInferAndRun(op_conf, scope_symbol=None):
    op_attribute = compile_ctx.CurJobAddOp(op_conf, scope_symbol)

    def BuildInstruction(builder):
        get_blob_register_scope = blob_register.BnInOp2BlobObjectScope
        with get_blob_register_scope(op_attribute) as bn_in_op2blob_object:
            builder.StatelessCall(
                op_attribute,
                scope_symbol.device_parallel_desc_symbol.parallel_conf,
                bn_in_op2blob_object=bn_in_op2blob_object,
            )

    vm_util.LogicalRun(BuildInstruction)
    return op_attribute
