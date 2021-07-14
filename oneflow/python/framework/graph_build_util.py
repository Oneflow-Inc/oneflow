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
from contextlib import contextmanager

from google.protobuf import text_format

import oneflow.core.job.scope_pb2 as scope_pb2_util
import oneflow.python.framework.attr_util as attr_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.runtime_mode as runtime_mode
import oneflow.python.framework.scope_util as scope_util
import oneflow.python.framework.session_context as session_context
import oneflow._oneflow_internal


lazy_mode = oneflow._oneflow_internal.lazy_mode


@contextmanager
def graph_build_context(config_proto, session):
    device_tag_and_ids = placement_util.GetDefaultMachineDeviceIds(session.resource)
    scope = scope_util.MakeInitialScope(
        config_proto,
        *device_tag_and_ids,
        None,  # TODO(): set hierarchy from user graph config
        False,  # is_mirrored
    )

    with lazy_mode.gard(True):
        with JobBuildAndInferCtx(config_proto):
            with scope_util.ScopeContext(scope):
                yield


class JobBuildAndInferCtx(object):
    def __init__(self, config_proto):
        self._job_conf = config_proto

    def __enter__(self):
        c_api_util.JobBuildAndInferCtx_Open(self._job_conf.job_name())
        c_api_util.CurJobBuildAndInferCtx_SetJobConf(self._job_conf)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # TODO(xuxiaoyu): open job optimization pass
            # oneflow._oneflow_internal.CurJobBuildAndInferCtx_Complete()
            oneflow._oneflow_internal.JobBuildAndInferCtx_Close()
            return True
        else:
            return False

class BlockScopeContext(object):
    def __init__(self, block):
        self._block = block 
        self._prev_scope = oneflow._oneflow_internal.GetCurrentScope()
        assert self._prev_scope is not None
        self._new_scope = _make_new_scope(self._prev_scope, self._block)

    def __enter__(self):
        print("enter block ", self._block.name_prefix + self._block.name)
        oneflow._oneflow_internal.GlobalScopeStackPush(self._new_scope)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            assert oneflow._oneflow_internal.GetCurrentScope() is self._new_scope 
            oneflow._oneflow_internal.GlobalScopeStackPop()
            assert oneflow._oneflow_internal.GetCurrentScope() is self._prev_scope 
            print("exit block ", self._block.name_prefix + self._block.name)
            return True
        else:
            return False

def _make_new_scope(prev_scope, block):
    assert prev_scope is not None
   
    attr_dict = dict()
    if block.config.stage_id is not None:
        attr_dict["pipeline_stage_id_hint"] = block.config.stage_id
    if block.config.activation_checkpointing is not None:
        attr_dict["checkpointing"] = block.config.activation_checkpointing

    name2default = session_context.GetDefaultSession().scope_attr_name2default_val
    def scope_proto_setter(scope_proto):
        # set attr
        for attr_name, py_value in attr_dict.items():
            assert attr_name in name2default
            attr_util.SetAttrValue(
                scope_proto.mutable_attr_name2attr_value()[attr_name],
                py_value,
                name2default[attr_name],
            )
        # append name prefix
        scope_proto.add_scope_op_name_prefixes(block.name)


    new_scope = None
    def BuildScope(builder):
        nonlocal new_scope
        new_scope = builder.BuildScopeByProtoSetter(
            prev_scope, scope_proto_setter
        )
        assert new_scope is not None

    oneflow._oneflow_internal.deprecated.LogicalRun(BuildScope)
    return new_scope

def scope_to_proto(scope):
    return text_format.Parse(scope._proto_str, scope_pb2_util.ScopeProto())