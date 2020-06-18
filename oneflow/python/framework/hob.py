import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.runtime_mode as rt_mode
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.lib.core.high_order_bool import HighOrderBool


def InRuntimeModeHOB(mode):
    assert rt_mode.IsValidMode(mode)
    return HighOrderBool(
        "Current mode is %s" % mode, lambda: rt_mode.CurrentMode() == mode
    )


in_normal_mode = InRuntimeModeHOB(rt_mode.NORMAL_MODE)
in_global_mode = InRuntimeModeHOB(rt_mode.GLOBAL_MODE)
in_device_mode = InRuntimeModeHOB(rt_mode.DEVICE_MODE)


def _IsEnvInitialized():
    assert in_normal_mode()
    return c_api_util.IsEnvInited()


env_initialized = HighOrderBool("Environment initialized", _IsEnvInitialized)


def _AnyGlobalFunctionDefined():
    assert in_normal_mode()
    return session_ctx.GetDefaultSession().AnyGlobalFunctionDefined()


any_global_function_defined = HighOrderBool(
    "Any global function defined", _AnyGlobalFunctionDefined
)


def _EagerExecutionEnabled():
    return c_api_util.EagerExecutionEnabled()


eager_execution_enabled = HighOrderBool(
    "Eager execution enabled ", _EagerExecutionEnabled
)


def _IsSessionInitialized():
    assert in_normal_mode()
    return session_ctx.GetDefaultSession().is_running


session_initialized = HighOrderBool("Session initialized", _IsSessionInitialized)


def _IsCurrentFunctionTrainable():
    assert in_global_mode()
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    return session_ctx.GetDefaultSession().GetFunctionDesc(job_name)


is_trainable = HighOrderBool(
    "Current global function is trainable", _IsCurrentFunctionTrainable
)


def _InPlacementScope():
    return len(session_ctx.GetDefaultSession().placement_scope_stack) > 0


in_placement_scope = HighOrderBool("In a placement scope", _InPlacementScope)
