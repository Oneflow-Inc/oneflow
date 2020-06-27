import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.runtime_mode as rt_mode
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.c_api_util as c_api_util
import oneflow
from oneflow.python.lib.core.high_order_bool import bool_functor


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
    return c_api_util.IsEnvInited()


@bool_functor("Any global function defined")
def any_global_function_defined(ctx):
    assert in_normal_mode(ctx)
    return session_ctx.GetDefaultSession().AnyGlobalFunctionDefined()


@bool_functor("Eager execution enabled")
def eager_execution_enabled(ctx):
    return c_api_util.EagerExecutionEnabled()


@bool_functor("Session initialized")
def session_initialized(ctx):
    assert in_normal_mode(ctx)
    return session_ctx.GetDefaultSession().is_running


@bool_functor("Current global function is trainable")
def is_trainable(ctx):
    assert in_global_mode(ctx)
    if c_api_util.EagerExecutionEnabled():
        return session_ctx.GetDefaultSession().CurrentEagerGlobalFunctionDesc()
    else:
        job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
        return session_ctx.GetDefaultSession().GetFunctionDesc(job_name)


@bool_functor("In a placement scope")
def in_placement_scope(ctx):
    return len(session_ctx.GetDefaultSession().placement_scope_stack) > 0


@bool_functor("Current placement is physical")
def is_current_placement_physical(ctx):
    return oneflow.placement.current_scope().is_physical_placement


@bool_functor("Current machine is master")
def is_current_machine_master(ctx):
    return c_api_util.CurrentMachineId() == 0


@bool_functor("Consistent view enabled")
def consistent_view_enabled(ctx):
    return oneflow.distribute.consistent_strategy_enabled()
