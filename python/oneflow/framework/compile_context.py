from contextlib import contextmanager

import oneflow
import oneflow._oneflow_internal
import oneflow.experimental.name_scope as name_scope
import oneflow.framework.c_api_util as c_api_util
import oneflow.framework.distribute_context as distribute_ctx
import oneflow.framework.hob as hob
import oneflow.framework.placement_context as placement_context
import oneflow.framework.session_context as session_ctx
import oneflow.lib.core.enable_if as enable_if


def GetCurJobConfigProto():
    return enable_if.unique([GetEagerCurJobConfigProto, GetLazyCurJobConfigProto])()


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def GetEagerCurJobConfigProto():
    function_desc = session_ctx.GetDefaultSession().CurrentEagerGlobalFunctionDesc()
    assert function_desc is not None
    return function_desc.job_config_proto


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def GetLazyCurJobConfigProto():
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    function_desc = session_ctx.GetDefaultSession().GetLazyFunctionDesc(job_name)
    assert function_desc is not None
    return function_desc.job_config_proto


logged_op_confs = set({})


def CurJobAddOp(op_conf, scope_symbol=None):
    if distribute_ctx.IsMirroredStrategyEnabled():
        return CurJobAddMirroredOp(op_conf, scope_symbol)
    return CurJobAddConsistentOp(op_conf, scope_symbol)


def CurJobAddConsistentOp(op_conf, scope_symbol=None):
    if scope_symbol is None:
        scope_symbol = oneflow.current_scope()
    op_conf.scope_symbol_id = scope_symbol.symbol_id
    if not op_conf.HasField("device_tag"):
        device_tag = scope_symbol.device_parallel_desc_symbol.device_tag
        op_conf.device_tag = device_tag
    op_attr = c_api_util.CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf)
    if c_api_util.IsInterfaceOpConf(op_conf):
        sess = session_ctx.GetDefaultSession()
        sess.AddInfo4InterfaceOpName(op_conf.name, op_attr)
    return op_attr


def CurJobAddMirroredOp(op_conf, scope_symbol=None):
    assert not hob.consistent_view_enabled(None)
    if scope_symbol is None:
        scope_symbol = oneflow.current_scope()
    op_conf.scope_symbol_id = scope_symbol.symbol_id
    if not op_conf.HasField("device_tag"):
        device_tag = scope_symbol.device_parallel_desc_symbol.device_tag
        op_conf.device_tag = device_tag
    op_attr = c_api_util.CurJobBuildAndInferCtx_AddAndInferMirroredOp(op_conf)
    if c_api_util.IsInterfaceOpConf(op_conf):
        sess = session_ctx.GetDefaultSession()
        sess.AddInfo4InterfaceOpName(op_conf.name, op_attr)
    return op_attr
