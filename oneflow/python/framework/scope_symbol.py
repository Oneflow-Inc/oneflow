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

from oneflow.python.eager.symbol import Symbol
import oneflow.python.eager.symbol_storage as symbol_storage
import oneflow_api.oneflow.core.job.scope as scope_cfg
import oneflow_api.oneflow.core.job.placement as placement_cfg
import oneflow_api
import collections
import re


class ScopeSymbol(Symbol):
    def __init__(self, symbol_id, scope_proto, parent_scope_symbol=None):
        Symbol.__init__(self, symbol_id, scope_proto)
        self.parent_scope_symbol_ = parent_scope_symbol
        self.job_desc_symbol_ = oneflow_api.GetJobConfSymbol(
            scope_proto.job_desc_symbol_id()
        )
        self.device_parallel_desc_symbol_ = oneflow_api.GetPlacementSymbol(
            scope_proto.device_parallel_desc_symbol_id()
        )
        self.host_parallel_desc_symbol_ = oneflow_api.GetPlacementSymbol(
            scope_proto.host_parallel_desc_symbol_id()
        )
        self.auto_increment_id_ = 0

    def auto_increment_id(self):
        self.auto_increment_id_ = self.auto_increment_id_ + 1
        return self.auto_increment_id_

    @property
    def session_id(self):
        return self.data.session_id()

    @property
    def job_desc_symbol(self):
        return self.job_desc_symbol_

    @property
    def device_parallel_desc_symbol(self):
        return self.device_parallel_desc_symbol_

    @property
    def parent_scope_symbol(self):
        return self.parent_scope_symbol_

    def BuildBySetter(self, instruction_builder, setter):
        scope_proto = self._CloneScopeProto()
        setter(scope_proto)
        return instruction_builder.GetScopeSymbol(scope_proto)

    def BuildWithNewParallelDesc(
        self, instruction_builder, device_tag, machine_device_ids
    ):
        if isinstance(machine_device_ids, str):
            machine_device_ids = [machine_device_ids]

        def SetScopeProto(scope_proto):
            parallel_conf = MakeParallelConf(device_tag, machine_device_ids)
            device_parallel_desc_sym = instruction_builder.GetParallelDescSymbol(
                parallel_conf
            )
            parallel_conf = MakeParallelConf("cpu", machine_device_ids)
            host_parallel_desc_sym = instruction_builder.GetParallelDescSymbol(
                parallel_conf
            )
            scope_proto.set_device_parallel_desc_symbol_id(
                device_parallel_desc_sym.symbol_id
            )
            scope_proto.set_host_parallel_desc_symbol_id(
                host_parallel_desc_sym.symbol_id
            )

        return self.BuildBySetter(instruction_builder, SetScopeProto)

    def BuildWithNewParallelConf(self, instruction_builder, parallel_conf):
        tag_and_dev_ids = oneflow_api.GetDeviceTagAndMachineDeviceIds(parallel_conf)
        return self.BuildWithNewParallelDesc(instruction_builder, *tag_and_dev_ids)

    def BuildWithNewIsMirrored(self, instruction_builder, is_mirrored):
        def SetScopeProto(scope_proto):
            if is_mirrored:
                scope_proto.mutable_opt_mirrored_parallel_conf().mutable_mirrored_parallel()
            else:
                scope_proto.mutable_opt_mirrored_parallel_conf().clear_mirrored_parallel()

        return self.BuildBySetter(instruction_builder, SetScopeProto)

    def BuildWithNewScopeName(self, instruction_builder, scope_name):
        def SetScopeProto(scope_proto):
            scope_proto.add_scope_op_name_prefixes(scope_name)

        return self.BuildBySetter(instruction_builder, SetScopeProto)

    def _CloneScopeProto(self):
        scope_proto = scope_cfg.ScopeProto()
        scope_proto.CopyFrom(self.data)
        return scope_proto


def BuildInitialScope(
    instruction_builder,
    session_id,
    job_conf,
    device_tag,
    machine_device_ids,
    is_mirrored,
):
    scope_proto = scope_cfg.ScopeProto()
    scope_proto.set_session_id(session_id)
    job_conf_sym = instruction_builder.GetJobConfSymbol(job_conf)
    scope_proto.set_job_desc_symbol_id(job_conf_sym.symbol_id)
    parallel_conf = MakeParallelConf(device_tag, machine_device_ids)
    device_parallel_desc_sym = instruction_builder.GetParallelDescSymbol(parallel_conf)
    scope_proto.set_device_parallel_desc_symbol_id(device_parallel_desc_sym.symbol_id)
    parallel_conf = MakeParallelConf("cpu", machine_device_ids)
    host_parallel_desc_sym = instruction_builder.GetParallelDescSymbol(parallel_conf)
    scope_proto.set_host_parallel_desc_symbol_id(host_parallel_desc_sym.symbol_id)
    if is_mirrored:
        scope_proto.mutable_opt_mirrored_parallel_conf().mutable_mirrored_parallel()
    else:
        scope_proto.mutable_opt_mirrored_parallel_conf().clear_mirrored_parallel()
    return instruction_builder.GetScopeSymbol(scope_proto)


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
