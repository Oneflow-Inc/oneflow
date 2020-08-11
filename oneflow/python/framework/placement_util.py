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
from __future__ import absolute_import
import re
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.hob as hob
from oneflow.python.oneflow_export import oneflow_export, oneflow_deprecate
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.eager.device_scope_stack as device_scope_stack
import oneflow
import traceback


@oneflow_export("placement.current_scope")
def api_current_placement_scope() -> placement_ctx.PlacementScope:
    print(
        "WARNING: oneflow.placement.current_scope has been deprecated. "
        "Please use oneflow.current_scope.device_parallel_desc_symbol instead."
    )
    print(traceback.format_stack()[-2])
    api = enable_if.unique(
        [global_mode_cur_placement_scope, normal_mode_cur_placement_scope]
    )
    return api()


@enable_if.condition(hob.in_global_mode & hob.in_placement_scope)
def global_mode_cur_placement_scope():
    return placement_ctx.PlacementScopeStackTop()


@enable_if.condition(hob.in_normal_mode)
def normal_mode_cur_placement_scope():
    return device_scope_stack.CurrentPlacement()


@oneflow_export("device_prior_placement", "fixed_placement")
@oneflow_deprecate()
def deprecated_placement(*args, **kwargs):
    print(
        "WARNING:",
        "oneflow.device_prior_placement/oneflow.fixed_placement",
        "will be removed in the future, use {} instead.".format(
            "oneflow.scope.placement"
        ),
    )
    print(traceback.format_stack()[-2])
    return api_placement(*args, **kwargs)


@oneflow_export("scope.placement")
def api_placement(
    device_tag: str, machine_device_ids: str
) -> placement_ctx.PlacementScope:
    from oneflow.python.compatibility import with_cuda

    if with_cuda == False:
        device_tag = "cpu"
    func = enable_if.unique([GetPlacementScope, GetNormalModePlacementScope])
    return func(device_tag, machine_device_ids)


@enable_if.condition(
    hob.in_global_mode
    | (hob.in_normal_mode & hob.env_initialized & ~hob.session_initialized)
)
def GetPlacementScope(device_tag, machine_device_ids):
    return placement_ctx.PlacementScope(device_tag, machine_device_ids)


@enable_if.condition(hob.in_normal_mode & hob.session_initialized)
def GetNormalModePlacementScope(device_tag, machine_device_ids):
    sess = session_ctx.GetDefaultSession()
    scope = sess.MakeScope(
        lambda old_scope, builder: old_scope.BuildWithNewParallelDesc(
            builder, device_tag, machine_device_ids
        )
    )
    return sess.NewCurrentScope(scope)


def GetDefaultMachineDeviceIds(resource):
    if resource.HasField("gpu_device_num") and resource.gpu_device_num > 0:
        return "gpu", placement_ctx.GetGpuMachineDeviceIds(resource)
    elif resource.HasField("cpu_device_num"):
        return "cpu", placement_ctx.GetCpuMachineDeviceIds(resource)
    else:
        raise NotImplementedError
