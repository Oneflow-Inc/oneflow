from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_ctx
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.blob_register as blob_register_util
import oneflow.python.eager.op_executor as op_executor
import oneflow

blob_register = blob_register_util.GetDefaultBlobRegister()


def Interpret(op_conf, scope_symbol=None):
    if scope_symbol is None:
        scope_symbol = oneflow.scope.current_scope()
    func = enable_if.unique([LazyInfer, EagerInferAndRun])
    return func(compile_ctx.CurJobAddOp, op_conf, scope_symbol)


def ConsistentInterpret(op_conf, scope_symbol=None):
    if scope_symbol is None:
        scope_symbol = oneflow.scope.current_scope()
    func = enable_if.unique([LazyInfer, EagerInferAndRun])
    return func(compile_ctx.CurJobAddConsistentOp, op_conf, scope_symbol)


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def LazyInfer(add_and_infer, op_conf, scope_symbol=None):
    return add_and_infer(op_conf, scope_symbol)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def EagerInferAndRun(add_and_infer, op_conf, scope_symbol=None):
    op_attribute = add_and_infer(op_conf, scope_symbol)
    parallel_conf = scope_symbol.device_parallel_desc_symbol.parallel_conf
    op_executor.Interpret(op_attribute, parallel_conf, blob_register)
    return op_attribute
