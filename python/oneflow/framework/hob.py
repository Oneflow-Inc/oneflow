import oneflow
import oneflow._oneflow_internal
import oneflow.framework.runtime_mode as rt_mode
import oneflow.framework.session_context as session_ctx
from oneflow.lib.core.high_order_bool import bool_functor


@bool_functor("Current mode is %s" % rt_mode.NORMAL_MODE)
def in_normal_mode(ctx):
    return rt_mode.CurrentMode() == rt_mode.NORMAL_MODE


@bool_functor("Current mode is %s" % rt_mode.GLOBAL_MODE)
def in_global_mode(ctx):
    return rt_mode.CurrentMode() == rt_mode.GLOBAL_MODE


@bool_functor("Current mode is %s" % rt_mode.DEVICE_MODE)
def in_device_mode(ctx):
    return rt_mode.CurrentMode() == rt_mode.DEVICE_MODE


@bool_functor("Environment initialized")
def env_initialized(ctx):
    assert in_normal_mode(ctx)
    return oneflow._oneflow_internal.IsEnvInited()


@bool_functor("Any global function defined")
def any_global_function_defined(ctx):
    assert in_normal_mode(ctx)
    return session_ctx.GetDefaultSession().AnyGlobalFunctionDefined()


@bool_functor("Eager execution enabled")
def eager_execution_enabled(ctx):
    return oneflow._oneflow_internal.EagerExecutionEnabled()


@bool_functor("Session initialized")
def session_initialized(ctx):
    assert in_normal_mode(ctx)
    return session_ctx.GetDefaultSession().is_running


@bool_functor("Current global function is trainable")
def is_trainable(ctx):
    assert in_global_mode(ctx)
    if oneflow._oneflow_internal.EagerExecutionEnabled():
        return session_ctx.GetDefaultSession().CurrentEagerGlobalFunctionDesc()
    else:
        job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
        return session_ctx.GetDefaultSession().GetFunctionDesc(job_name)


@bool_functor("Current machine is master")
def is_current_machine_master(ctx):
    return oneflow._oneflow_internal.CurrentMachineId() == 0


@bool_functor("Consistent view enabled")
def consistent_view_enabled(ctx):
    return oneflow.scope.consistent_view_enabled()


@bool_functor("Mirrored view enabled")
def mirrored_view_enabled(ctx):
    return oneflow.scope.mirrored_view_enabled()
