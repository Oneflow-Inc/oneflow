from __future__ import absolute_import

import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.ops.op_util as op_util
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('placement.current_scope')
def cur_placement_scope():
    return placement_ctx.PlacementScopeStackTop()

@oneflow_export('fixed_placement')
class FixedPlacementScope(placement_ctx.PlacementScope):
    def __init__(self, device_tag, machine_device_ids):
        placement_ctx.PlacementScope.__init__(self, device_tag, machine_device_ids)

    def GetDeviceTag4OpConf(self, op_conf): return self.default_device_tag

@oneflow_export('device_prior_placement')
class DevicePriorPlacementScope(placement_ctx.PlacementScope):
    def __init__(self, device_tag, machine_device_ids):
        placement_ctx.PlacementScope.__init__(self, device_tag, machine_device_ids)

    def GetDeviceTag4OpConf(self, op_conf):
        if op_util.IsOpConfOnlyCpuSupported(op_conf): return "cpu"
        return self.default_device_tag

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
