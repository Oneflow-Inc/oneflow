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
import oneflow.python.lib.core.scope_stack as scope_stack
import oneflow.python.lib.core.lazy as lazy
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.hob as hob
from oneflow.python.oneflow_export import oneflow_export
import re
import oneflow


class PhysicalPlacementScope(placement_ctx.PlacementScope):
    def __init__(self, device_tag, machine_device_ids):
        placement_ctx.PlacementScope.__init__(self, device_tag, machine_device_ids)
        self.physical_symbol_id_ = c_api_util.NewPhysicalSymbolId()

    @property
    def is_physical_placement(self):
        return True

    @property
    def physical_symbol_id(self):
        return self.physical_symbol_id_


def CurrentPlacement():
    global device_scope_stack
    return device_scope_stack.Current().value


@enable_if.condition(hob.in_normal_mode & hob.env_initialized)
def NewPlacementScope(device_tag_and_id):
    assert type(device_tag_and_id) is str
    assert re.match("^(cpu)|(gpu):\d+$", device_tag_and_id) is not None
    device_tag, device_id = device_tag_and_id.split(":")

    def GetPhysicalPlacementScope():
        return _NewPhysicalPlacementScope(device_tag, device_id)

    global device_scope_stack
    return device_scope_stack.NewScope(lazy.Lazy(GetPhysicalPlacementScope))


@oneflow_export("device")
def api_device(device_tag_and_id):
    return enable_if.unique([NewPlacementScope])(device_tag_and_id)


@enable_if.condition(
    hob.in_normal_mode & hob.env_initialized & hob.is_current_machine_master
)
def EagerPlacementScope(device_tag, machine_device_ids):
    def EagerPlacementScope():
        return placement_ctx.PlacementScope(device_tag, machine_device_ids)

    global device_scope_stack
    return device_scope_stack.NewScope(lazy.Lazy(EagerPlacementScope))


@oneflow_export("eager_fixed_placement")
def api_eager_fixed_placement(device_tag, machine_device_ids):
    return enable_if.unique([EagerPlacementScope])(device_tag, machine_device_ids)


def _GetInitDeviceScope():
    resource = oneflow.env.current_resource()
    if resource.HasField("gpu_device_num"):
        device_tag = "gpu"
    elif resource.HasField("cpu_device_num"):
        device_tag = "cpu"
    else:
        raise NotImplementedError
    return _NewPhysicalPlacementScope(device_tag, 0)


def _NewPhysicalPlacementScope(device_tag, device_id):
    current_machine_id = oneflow.current_machine_id()
    return PhysicalPlacementScope(device_tag, "%d:%s" % (current_machine_id, device_id))


device_scope_stack = scope_stack.ScopeStack(lazy.Lazy(_GetInitDeviceScope))
