from oneflow.python.lib.core.high_order_bool import HighOrderBool
import oneflow.python.framework.runtime_mode as rt_mode
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.g_func_ctx as g_func_ctx
import oneflow.python.framework.c_api_util as c_api_util

def InRuntimeModeHOB(mode):
    assert rt_mode.IsValidMode(mode)
    return HighOrderBool("Current mode is %s" % mode, lambda: rt_mode.CurrentMode() == mode)

in_normal_mode = InRuntimeModeHOB(rt_mode.NORMAL_MODE)
in_global_mode = InRuntimeModeHOB(rt_mode.GLOBAL_MODE)
in_device_mode = InRuntimeModeHOB(rt_mode.DEVICE_MODE)

def _IsEnvInitialized():
    assert in_normal_mode()
    return c_api_util.IsEnvInited()

env_initialized = HighOrderBool("Environment initialized", _IsEnvInitialized)

def _IsSessionInitialized():
    assert in_normal_mode()
    return session_ctx.GetDefaultSession().is_running

session_initialized = HighOrderBool("Session initialized", _IsSessionInitialized)

def _IsCurrentFunctionTrainable():
    assert in_global_mode()
    job_name = g_func_ctx.JobBuildAndInferCtx_GetCurrentJobName()
    return session_ctx.GetDefaultSession().GetFunctionDesc(job_name)

is_trainable = HighOrderBool("Current global function is trainable", _IsCurrentFunctionTrainable)
