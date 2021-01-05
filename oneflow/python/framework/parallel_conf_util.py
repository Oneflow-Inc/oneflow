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
import re
import oneflow_api.oneflow.core.job.placement as placement_cfg


def GetDeviceTagAndMachineDeviceIds(parallel_conf):
    machine_device_ids = []
    for device_name in list(parallel_conf.device_name()):
        machine_device_ids.append(device_name)
    device_tag = parallel_conf.device_tag()
    return device_tag, machine_device_ids


def MakeParallelConf(device_tag, machine_device_ids):
    assert isinstance(machine_device_ids, (list, tuple))

    parallel_conf = placement_cfg.ParallelConf()
    parallel_conf.set_device_tag(device_tag)
    for machine_device_id in machine_device_ids:
        assert isinstance(
            machine_device_id, str
        ), "type of machine_device_id (%s) is not string" % type(machine_device_id)
        assert re.match("^\d+:\d+(-\d+)?$", machine_device_id) is not None, (
            "machine_device_id: %s is not valid" % machine_device_id
        )
        parallel_conf.add_device_name(machine_device_id)

    return parallel_conf
