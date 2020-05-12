from __future__ import absolute_import
import re
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.hob as hob
import oneflow.python.ops.op_util as op_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow

@oneflow_export('placement.current_scope', enable_if=hob.in_placement_scope)
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

class PhysicalPlacementScope(FixedPlacementScope):
    def __init__(self, device_tag, machine_device_ids):
        FixedPlacementScope.__init__(self, device_tag, machine_device_ids)
        self.physical_symbol_id_ = oneflow.vm.new_physical_symbol_id()

    @property
    def physical_symbol_id(self): return self.physical_symbol_id_

@oneflow_export('device', enable_if=hob.in_normal_mode & hob.env_initialized)
def DeviceScope(device_tag_and_id):
    assert type(device_tag_and_id) is str
    assert re.match("^(cpu)|(gpu):\d+$", device_tag_and_id) is not None
    current_machine_id = oneflow.current_machine_id()
    device_tag, device_id = device_tag_and_id.split(":")
    return PhysicalPlacementScope(device_tag, "%d:%s" % (current_machine_id, device_id))

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
