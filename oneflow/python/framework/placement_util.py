from __future__ import absolute_import

import re
import oneflow.core.job.placement_pb2 as placement_proto
import oneflow.python.framework.placement_context as placement_context
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.ops.op_util as op_util
import oneflow.python.framework.device_util as device_util
import oneflow.python.framework.job_builder as job_builder
from oneflow.python.oneflow_export import oneflow_export

class PlacementScope(object):
    def __init__(self, device_tag, machine_device_ids):
        self.device_tag_ = device_tag
        self.machine_device_ids_ = machine_device_ids
        self.op_confs_ = []

    def AppendOpConf(self, op_conf):
        self.op_confs_.append(op_conf)

    def GetDeviceType4OpConf(self, op_conf):
        return device_util.DeviceType4DeviceTag(self.GetDeviceTag4OpConf(op_conf))

    def GetDeviceTag4OpConf(self, op_conf):
        raise NotImplementedError

    @property
    def default_device_tag(self): return self.device_tag_

    @property
    def machine_device_ids(self): return self.machine_device_ids_

    @property
    def op_confs(self): return self.op_confs_

    def __enter__(self):
        placement_context.PlacementScopeStackPush(self)

    def ParallelConfAndOpNames(self):
        raise NotImplementedError
        
    def __exit__(self, *args):
        assert self == placement_context.PlacementScopeStackPop()
        for parallel_conf, op_names in self.ParallelConfAndOpNames():
            placement_group = placement_proto.PlacementGroup()
            placement_group.op_set.op_name.extend(op_names)
            placement_group.parallel_conf.CopyFrom(parallel_conf)
            job_builder.CurCtxAddPlacementGroup(placement_group)
            compile_context.cur_job.placement.placement_group.add().CopyFrom(placement_group)

@oneflow_export('fixed_placement')
class FixedPlacementScope(PlacementScope):
    def __init__(self, device_tag, machine_device_ids):
        PlacementScope.__init__(self, device_tag, machine_device_ids)

    def GetDeviceTag4OpConf(self, op_conf): return self.default_device_tag

    def ParallelConfAndOpNames(self):
        parallel_conf = MakeParallelConf(self.default_device_tag, self.machine_device_ids)
        yield parallel_conf, map(lambda op_conf: op_conf.name, self.op_confs)
        
@oneflow_export('device_prior_placement')
class DevicePriorPlacementScope(PlacementScope):
    def __init__(self, device_tag, machine_device_ids):
        PlacementScope.__init__(self, device_tag, machine_device_ids)

    def GetDeviceTag4OpConf(self, op_conf):
        if op_util.IsOpConfOnlyCpuSupported(op_conf): return "cpu"
        return self.default_device_tag

    def ParallelConfAndOpNames(self):
        device_tag2op_names = {}
        for op_conf in self.op_confs:
            device_tag = self.GetDeviceTag4OpConf(op_conf)
            if device_tag not in device_tag2op_names: device_tag2op_names[device_tag] = []
            device_tag2op_names[device_tag].append(op_conf.name)
        for device_tag, op_names in device_tag2op_names.items():
            if len(op_names) == 0: continue
            parallel_conf = MakeParallelConf(device_tag, self.machine_device_ids)
            yield parallel_conf, op_names
        
def MakeParallelConf(device_tag, machine_device_ids):
    if isinstance(machine_device_ids, str): machine_device_ids = [machine_device_ids]
    device_names = []
    for machine_device_id in machine_device_ids:
        assert isinstance(machine_device_id, str)
        assert re.match("^\d+:\d+(-\d+)?$", machine_device_id) is not None
        pair = machine_device_id.split(':')
        device_names.append("%s:%s:%s" % (pair[0], device_tag, pair[1]))
    parallel_conf = placement_proto.ParallelConf()
    parallel_conf.policy = placement_proto.kDataParallel
    parallel_conf.device_name.extend(device_names)
    return parallel_conf

def GetDefaultMachineDeviceIds(resource):
    if resource.HasField('gpu_device_num'):
        return 'gpu', GetGpuDefaultMachineDeviceIds(resource)
    elif resource.HasField('cpu_device_num'):
        return 'cpu', GetCpuDefaultMachineDeviceIds(resource)
    else:
        raise NotImplementedError

def GetGpuDefaultMachineDeviceIds(resource):
    assert len(resource.machine) > 0
    assert resource.HasField('gpu_device_num')
    return ["%s:0-%s" % (m.id, resource.gpu_device_num - 1) for m in resource.machine]
    
def GetCpuDefaultMachineDeviceIds(resource):
    assert len(resource.machine) > 0
    assert resource.HasField('cpu_device_num')
    return ["%s:0-%s" % (m.id, resource.cpu_device_num - 1) for m in resource.machine]
