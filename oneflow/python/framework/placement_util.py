from __future__ import absolute_import

import re
import oneflow.core.job.placement_pb2 as placement_proto
import oneflow.python.framework.placement_context as placement_context

def MakeParallelConf(device_names):
    if isinstance(device_names, str): device_names = [device_names]
    for dev_name in device_names:
        assert isinstance(dev_name, str)
        assert re.match("^\d+:(cpu|gpu):\d+(-\d+)?$", dev_name) is not None
    parallel_conf = placement_proto.ParallelConf()
    parallel_conf.policy = placement_proto.kDataParallel
    parallel_conf.device_name.extend(device_names)
    return parallel_conf

def GetJobPlacementParallelConf(job_name, resource):
    if job_name in placement_context.job_name2default_parallel_conf:
        return placement_context.job_name2default_parallel_conf[job_name]
    elif resource.HasField('gpu_device_num'):
        return MakeDefaultGpuParallelConf(resource)
    elif resource.HasField('cpu_device_num'):
        return MakeDefaultCpuParallelConf(resource)
    else:
        raise NotImplementedError

def MakeDefaultGpuParallelConf(resource):
    assert len(resource.machine) > 0
    assert resource.HasField('gpu_device_num')
    parallel_conf = placement_proto.ParallelConf()
    parallel_conf.policy = placement_proto.kDataParallel
    for m in resource.machine:
        parallel_conf.device_name.extend(["%s:gpu:0-%s" %(m.id, resource.gpu_device_num - 1)])
    return parallel_conf
    
def MakeDefaultCpuParallelConf(resource):
    assert len(resource.machine) > 0
    assert resource.HasField('cpu_device_num')
    parallel_conf = placement_proto.ParallelConf()
    parallel_conf.policy = placement_proto.kDataParallel
    for m in resource.machine:
        parallel_conf.device_name.extend(["%s:cpu:0-%s" %(m.id, resource.cpu_device_num - 1)])
    return parallel_conf

class PlacementScope(object):
    def __init__(self, parallel_conf):
        self.parallel_conf_ = parallel_conf

    def __enter__(self):
        placement_context.ParallelConfStackPush(self.parallel_conf_)
        
    def __exit__(self, *args):
        placement_context.ParallelConfStackPop()
