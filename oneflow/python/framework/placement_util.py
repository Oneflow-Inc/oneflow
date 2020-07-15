from __future__ import absolute_import

import oneflow.python.framework.hob as hob
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("placement.current_scope")
def api_placement_current_scope() -> placement_ctx.PlacementScope:
    return enable_if.unique([placement_current_scope])()


@enable_if.condition(hob.in_global_mode & hob.in_placement_scope)
def placement_current_scope():
    return placement_ctx.PlacementScopeStackTop()


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
        return "gpu", _GetGpuDefaultMachineDeviceIds(resource)
    elif resource.HasField("cpu_device_num"):
        return "cpu", _GetCpuDefaultMachineDeviceIds(resource)
    else:
        raise NotImplementedError


def _GetGpuDefaultMachineDeviceIds(resource):
    assert resource.machine_num > 0
    assert resource.HasField("gpu_device_num")
    return [
        "%s:0-%s" % (m_id, resource.gpu_device_num - 1)
        for m_id in range(resource.machine_num)
    ]


def _GetCpuDefaultMachineDeviceIds(resource):
    assert resource.machine_num > 0
    assert resource.HasField("cpu_device_num")
    return [
        "%s:0-%s" % (m_id, resource.gpu_device_num - 1)
        for m_id in range(resource.machine_num)
    ]
