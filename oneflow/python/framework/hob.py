"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow.python.framework.runtime_mode as rt_mode
import oneflow.python.framework.session_context as session_ctx
import oneflow
from oneflow.python.lib.core.high_order_bool import bool_functor
import oneflow_api


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
    return oneflow_api.IsEnvInited()


@bool_functor("Any global function defined")
def any_global_function_defined(ctx):
    assert in_normal_mode(ctx)
    return session_ctx.GetDefaultSession().AnyGlobalFunctionDefined()


@bool_functor("Eager execution enabled")
def eager_execution_enabled(ctx):
    return oneflow_api.EagerExecutionEnabled()


@bool_functor("Session initialized")
def session_initialized(ctx):
    assert in_normal_mode(ctx)
    return session_ctx.GetDefaultSession().is_running


@bool_functor("Current global function is trainable")
def is_trainable(ctx):
    assert in_global_mode(ctx)
    if oneflow_api.EagerExecutionEnabled():
        return session_ctx.GetDefaultSession().CurrentEagerGlobalFunctionDesc()
    else:
        job_name = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
        return session_ctx.GetDefaultSession().GetFunctionDesc(job_name)


@bool_functor("Current machine is master")
def is_current_machine_master(ctx):
    return oneflow_api.CurrentMachineId() == 0


@bool_functor("Consistent view enabled")
def consistent_view_enabled(ctx):
    return oneflow.scope.consistent_view_enabled()


@bool_functor("Mirrored view enabled")
def mirrored_view_enabled(ctx):
    return oneflow.scope.mirrored_view_enabled()
