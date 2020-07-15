from __future__ import absolute_import
import re
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.hob as hob
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.eager.device_scope_stack as device_scope_stack
import oneflow


@oneflow_export("placement.current_scope")
def api_current_placement_scope() -> placement_ctx.PlacementScope:
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


def api_fixed_placement(
    device_tag: str, machine_device_ids: str
) -> placement_ctx.FixedPlacementScope:
    return enable_if.unique([GetFixedPlacementScope])(device_tag, machine_device_ids)


@enable_if.condition(
    hob.in_global_mode
    | (hob.in_normal_mode & hob.env_initialized & ~hob.session_initialized)
)
def GetFixedPlacementScope(device_tag, machine_device_ids):
    return placement_ctx.FixedPlacementScope(device_tag, machine_device_ids)


@oneflow_export("device_prior_placement", "fixed_placement")
def deprecated_placement(*args, **kwargs):
    print(
        "WARNING:",
        "/".join(deprecated_placement._ONEFLOW_API),
        "will be removed in the future, use oneflow.scope.placement instead.",
    )
    return api_placement(*args, **kwargs)


@oneflow_export("scope.placement")
def api_placement(
    device_tag: str, machine_device_ids: str
) -> placement_ctx.DevicePriorPlacementScope:
    return enable_if.unique([GetDevicePriorPlacementScope])(
        device_tag, machine_device_ids
    )


@enable_if.condition(
    hob.in_global_mode
    | (hob.in_normal_mode & hob.env_initialized & ~hob.session_initialized)
)
def GetDevicePriorPlacementScope(device_tag, machine_device_ids):
    return placement_ctx.DevicePriorPlacementScope(device_tag, machine_device_ids)


def GetDefaultMachineDeviceIds(resource):
    if resource.HasField("gpu_device_num"):
        return "gpu", placement_ctx.GetGpuMachineDeviceIds(resource)
    elif resource.HasField("cpu_device_num"):
        return "cpu", placement_ctx.GetCpuMachineDeviceIds(resource)
    else:
        raise NotImplementedError
