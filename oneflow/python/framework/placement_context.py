from __future__ import absolute_import

import re
import collections
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.device_util as device_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.core.job.placement_pb2 as placement_proto_pb

class PlacementScope(object):
    def __init__(self, device_tag, machine_device_ids):
        self.device_tag_ = device_tag
        if isinstance(machine_device_ids, (list, tuple)) == False:
            machine_device_ids = [machine_device_ids]
        self.machine_device_ids_ = machine_device_ids
        self.default_parallel_conf_ = _MakeParallelConf(self.device_tag_, self.machine_device_ids_)
        self.machine_id2device_id_list_ = MakeMachineId2DeviceIdList(self.default_parallel_conf_)
        self.parallel_size_ = GetParallelSize(self.machine_id2device_id_list_)

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
        PlacementScopeStackPush(self)

    def __exit__(self, *args):
        assert self == PlacementScopeStackPop()

def PlacementScopeStackPush(placement_policy):
    session_ctx.GetDefaultSession().placement_scope_stack.insert(0, placement_policy)

def PlacementScopeStackPop():
    return session_ctx.GetDefaultSession().placement_scope_stack.pop(0)
    
def PlacementScopeStackTop():
    assert len(session_ctx.GetDefaultSession().placement_scope_stack) > 0, "no placement scope found"
    return session_ctx.GetDefaultSession().placement_scope_stack[0]
    
def CurPlacementGroupGetDeviceType(op_conf):
    assert len(session_ctx.GetDefaultSession().placement_scope_stack) > 0
    return session_ctx.GetDefaultSession().placement_scope_stack[0].GetDeviceType4OpConf(op_conf)

def ParallelConf4OpConf(op_conf):
    assert len(session_ctx.GetDefaultSession().placement_scope_stack) > 0
    return session_ctx.GetDefaultSession().placement_scope_stack[0].ParallelConf4OpConf(op_conf)

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

    parallel_conf = placement_proto_pb.ParallelConf()
    parallel_conf.device_name.extend(device_names)
    return parallel_conf

def MakeMachineId2DeviceIdList(parallel_conf):
    parallel_conf_str = str(parallel_conf)
    if parallel_conf_str not in _parallel_conf_str2ofrecord:
        ofrecord = c_api_util.GetMachine2DeviceIdListOFRecordFromParallelConf(parallel_conf)
        _parallel_conf_str2ofrecord[parallel_conf_str] = \
                {int(k) : list(v.int32_list.value) for k, v in ofrecord.feature.items()}
    return _parallel_conf_str2ofrecord[parallel_conf_str]

def GetParallelSize(key2list):
    size = 0
    for k, v in key2list.items(): size += len(v)
    return size

_parallel_conf_str2ofrecord = {}
