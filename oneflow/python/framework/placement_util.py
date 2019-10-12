from __future__ import absolute_import

import re
import oneflow.core.job.placement_pb2 as placement_proto
import oneflow.python.framework.placement_context as placement_context
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.ops.op_util as op_util
import oneflow.python.framework.device_util as device_util
import oneflow.python.framework.job_builder as job_builder
import oneflow.python.framework.c_api_util as c_api_util
import collections
from oneflow.python.oneflow_export import oneflow_export

class PlacementScope(object):
    def __init__(self, device_tag, machine_device_ids):
        self.device_tag_ = device_tag
        if isinstance(machine_device_ids, (list, tuple)) == False:
            machine_device_ids = [machine_device_ids]
        self.machine_device_ids_ = machine_device_ids
        self.default_parallel_conf_ = _MakeParallelConf(self.device_tag_, self.machine_device_ids_)
        self.machine_id2device_id_list_ = _MakeMachineId2DeviceIdList(self.default_parallel_conf_)
        self.parallel_size_ = _GetParallelSize(self.machine_id2device_id_list_)

    @property
    def default_device_tag(self): return self.device_tag_
    
    @property
    def default_parallel_conf(self): return self.default_parallel_conf_

    @property
    def machine_id2device_id_list(self): return self.machine_id2device_id_list_

    @property
    def parallel_size(self): return self.parallel_size_

    def ParallelConf4OpConf(self, op_conf):
        return _MakeParallelConf(self.GetDeviceTag4OpConf(op_conf), self.machine_device_ids_)

    def GetDeviceType4OpConf(self, op_conf):
        return device_util.DeviceType4DeviceTag(self.GetDeviceTag4OpConf(op_conf))

    def GetDeviceTag4OpConf(self, op_conf):
        raise NotImplementedError

    def __enter__(self):
        placement_context.PlacementScopeStackPush(self)

    def __exit__(self, *args):
        assert self == placement_context.PlacementScopeStackPop()

@oneflow_export('current_placement_scope.parallel_size')
def cur_placement_scope_parallel_num():
    return placement_context.PlacementScopeStackTop().parallel_size

@oneflow_export('fixed_placement')
class FixedPlacementScope(PlacementScope):
    def __init__(self, device_tag, machine_device_ids):
        PlacementScope.__init__(self, device_tag, machine_device_ids)

    def GetDeviceTag4OpConf(self, op_conf): return self.default_device_tag

@oneflow_export('device_prior_placement')
class DevicePriorPlacementScope(PlacementScope):
    def __init__(self, device_tag, machine_device_ids):
        PlacementScope.__init__(self, device_tag, machine_device_ids)

    def GetDeviceTag4OpConf(self, op_conf):
        if op_util.IsOpConfOnlyCpuSupported(op_conf): return "cpu"
        return self.default_device_tag

def _MakeParallelConf(device_tag, machine_device_ids):
    assert isinstance(machine_device_ids, collections.Sized)
    device_names = []
    for machine_device_id in machine_device_ids:
        assert isinstance(machine_device_id, str), \
            "type of machine_device_id (%s) is not string" % type(machine_device_id)
        assert re.match("^\d+:\d+(-\d+)?$", machine_device_id) is not None, \
            "machine_device_id: %s is not valid" % machine_device_id
        pair = machine_device_id.split(':')
        device_names.append("%s:%s:%s" % (pair[0], device_tag, pair[1]))

    parallel_conf = placement_proto.ParallelConf()
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

def _MakeMachineId2DeviceIdList(parallel_conf):
    parallel_conf_str = str(parallel_conf)
    if parallel_conf_str not in _parallel_conf_str2ofrecord:
        of_record = c_api_util.GetMachine2DeviceIdListOFRecordFromParallelConf(parallel_conf)
        _parallel_conf_str2ofrecord[parallel_conf_str] = of_record
    return _parallel_conf_str2ofrecord[parallel_conf_str]

def _GetParallelSize(ofrecord):
    size = 0
    for k, v in ofrecord.feature.items(): size += len(v.int32_list.value)
    return size

_parallel_conf_str2ofrecord = {}
