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
import traceback
from contextlib import contextmanager

from google.protobuf import text_format

import oneflow._oneflow_internal
import oneflow._oneflow_internal.oneflow.core.job.job_conf as job_conf_cfg
import oneflow.core.job.scope_pb2 as scope_pb2_util
import oneflow.framework.attr_util as attr_util
import oneflow.framework.session_context as session_ctx
from oneflow import oneflow_deprecate


def api_scope_config(**kwargs):
    name2default = session_ctx.GetDefaultSession().scope_attr_name2default_val

    def SetScopeProto(scope_proto):
        for (attr_name, py_value) in kwargs.items():
            assert attr_name in name2default
            attr_util.SetAttrValue(
                scope_proto.mutable_attr_name2attr_value()[attr_name],
                py_value,
                name2default[attr_name],
            )

    sess = session_ctx.GetDefaultSession()
    scope = MakeScope(
        lambda old_scope, builder: builder.BuildScopeByProtoSetter(
            old_scope, SetScopeProto
        )
    )
    return ScopeContext(scope)


def current_scope():
    """ Return current scope
    """
    return oneflow._oneflow_internal.GetCurrentScope()


from oneflow import oneflow_deprecate


def MakeScope(build_func):
    scope = None
    old_scope = oneflow._oneflow_internal.GetCurrentScope()
    assert old_scope is not None

    def BuildScope(builder):
        nonlocal scope
        scope = build_func(old_scope, builder)
        assert scope is not None

    oneflow._oneflow_internal.deprecated.PhysicalRun(BuildScope)
    return scope


def MakeInitialScope(job_conf, device_tag, machine_device_ids, hierarchy, is_mirrored):
    scope = None

    def BuildInitialScope(builder):
        nonlocal scope
        session_id = session_ctx.GetDefaultSession().id
        scope = builder.BuildInitialScope(
            session_id, job_conf, device_tag, machine_device_ids, hierarchy, is_mirrored
        )

    oneflow._oneflow_internal.deprecated.PhysicalRun(BuildInitialScope)
    return scope


@contextmanager
def ScopeContext(scope):
    old_scope = oneflow._oneflow_internal.GetCurrentScope()
    oneflow._oneflow_internal.GlobalScopeStackPush(scope)
    try:
        yield
    finally:
        assert oneflow._oneflow_internal.GetCurrentScope() is scope
        oneflow._oneflow_internal.GlobalScopeStackPop()
        assert oneflow._oneflow_internal.GetCurrentScope() is old_scope
