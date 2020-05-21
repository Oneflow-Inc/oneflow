from __future__ import absolute_import
import re
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.hob as hob
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.eager.device_scope_stack as device_scope_stack
import oneflow

@oneflow_export('placement.current_scope', enable_if=hob.in_global_mode & hob.in_placement_scope)
def global_mode_cur_placement_scope():
    return placement_ctx.PlacementScopeStackTop()


@oneflow_export('placement.current_scope', enable_if=hob.in_normal_mode)
def normal_mode_cur_placement_scope():
    return device_scope_stack.CurrentPlacement()

hob_before_session = (hob.in_normal_mode & hob.env_initialized & ~hob.session_initialized)
@oneflow_export('fixed_placement', enable_if=hob.in_global_mode | hob_before_session)
def GetFixedPlacementScope(device_tag, machine_device_ids):
    return placement_ctx.FixedPlacementScope(device_tag, machine_device_ids)


@oneflow_export('device_prior_placement')
def GetDevicePriorPlacementScope(device_tag, machine_device_ids):
    return placement_ctx.DevicePriorPlacementScope(device_tag, machine_device_ids)


def GetDefaultMachineDeviceIds(resource):
    if resource.HasField('gpu_device_num'):
        return 'gpu', _GetGpuDefaultMachineDeviceIds(resource)
    elif resource.HasField('cpu_device_num'):
        return 'cpu', _GetCpuDefaultMachineDeviceIds(resource)
    else:
        raise NotImplementedError


def _GetGpuDefaultMachineDeviceIds(resource):
    assert resource.machine_num > 0
    assert resource.HasField('gpu_device_num')
    return ["%s:0-%s" % (m_id, resource.gpu_device_num - 1) for m_id in range(resource.machine_num)]


def _GetCpuDefaultMachineDeviceIds(resource):
    assert resource.machine_num > 0
    assert resource.HasField('cpu_device_num')
    return ["%s:0-%s" % (m_id, resource.gpu_device_num - 1) for m_id in range(resource.machine_num)]
