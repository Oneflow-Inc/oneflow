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

import collections
import re

import oneflow.core.job.placement_pb2 as placement_pb
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.op_util as op_util
import oneflow.python.framework.session_context as session_ctx
import oneflow
import oneflow_api.oneflow.core.job.placement as placement_cfg
import oneflow_api


class PlacementScope(object):
    pass


class EmptyPlacementScope(PlacementScope):
    def __init__(self, device_tag, machine_device_ids, parallel_hierarchy):
        if isinstance(machine_device_ids, (list, tuple)) == False:
            machine_device_ids = [machine_device_ids]
        if (
            parallel_hierarchy is not None
            and isinstance(parallel_hierarchy, (list, tuple)) == False
        ):
            parallel_hierarchy = [parallel_hierarchy]
        self.device_tag_ = device_tag
        self.machine_device_ids_ = machine_device_ids
        self.parallel_hierarchy_ = parallel_hierarchy

    @property
    def device_tag(self):
        return self.device_tag_

    @property
    def machine_device_ids(self):
        return self.machine_device_ids_

    @property
    def parallel_hierarchy(self):
        return self.parallel_hierarchy_

    def __enter__(self):
        # do nothing
        pass

    def __exit__(self, *args):
        # do nothing
        pass


class GlobalModePlacementScope(PlacementScope):
    def __init__(self, scope_ctx):
        self.scope_ctx_ = scope_ctx

    def __enter__(self):
        self.scope_ctx_.__enter__()

    def __exit__(self, *args):
        self.scope_ctx_.__exit__(*args)


def MakeParallelConf4Resource(device_tag, resource):
    if device_tag == "gpu":
        assert resource.HasField("gpu_device_num")
        machine_device_ids = GetGpuMachineDeviceIds(resource)
    elif device_tag == "cpu":
        assert resource.HasField("cpu_device_num")
        machine_device_ids = GetCpuMachineDeviceIds(resource)
    else:
        raise NotImplementedError
    return oneflow_api.MakeParallelConf(device_tag, machine_device_ids)


def MakeMachineId2DeviceIdList(parallel_conf):
    parallel_conf_str = str(parallel_conf)
    global _parallel_conf_str2ofrecord
    if parallel_conf_str not in _parallel_conf_str2ofrecord:
        ofrecord = c_api_util.GetMachine2DeviceIdListOFRecordFromParallelConf(
            parallel_conf
        )
        _parallel_conf_str2ofrecord[parallel_conf_str] = {
            int(k): list(v.int32_list.value) for k, v in ofrecord.feature.items()
        }
    return _parallel_conf_str2ofrecord[parallel_conf_str]


def GetParallelSize(key2list):
    size = 0
    for k, v in key2list.items():
        size += len(v)
    return size


def GetGpuMachineDeviceIds(resource):
    assert resource.machine_num > 0
    assert resource.HasField("gpu_device_num")
    return [
        "%s:0-%s" % (m_id, resource.gpu_device_num - 1)
        for m_id in range(resource.machine_num)
    ]


def GetCpuMachineDeviceIds(resource):
    assert resource.machine_num > 0
    assert resource.HasField("cpu_device_num")
    return [
        "%s:0-%s" % (m_id, resource.cpu_device_num - 1)
        for m_id in range(resource.machine_num)
    ]


_parallel_conf_str2ofrecord = {}
