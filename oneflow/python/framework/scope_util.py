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
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.attr_util as attr_util
import oneflow_api.oneflow.core.job.job_conf as job_conf_cfg
from contextlib import contextmanager
from oneflow.python.oneflow_export import oneflow_export, oneflow_deprecate
import oneflow_api


@oneflow_export("experimental.scope.config")
def api_scope_config(**kwargs):
    name2default = session_ctx.GetDefaultSession().scope_attr_name2default_val

    def SetScopeProto(scope_proto):
        for attr_name, py_value in kwargs.items():
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


@oneflow_export("current_scope")
def api_current_scope():
    r""" Return current scope
    """
    return oneflow_api.GetCurrentScope()


@oneflow_export("scope.current_scope")
@oneflow_deprecate()
def deprecated_current_scope(*args, **kwargs):
    print(
        "WARNING:",
        "oneflow.scope.current_scope",
        "will be removed in the future, use {} instead.".format(
            "oneflow.current_scope"
        ),
    )
    print(traceback.format_stack()[-2])

    return api_current_scope(*args, **kwargs)


def MakeScope(build_func):
    scope = None
    old_scope = oneflow_api.GetCurrentScope()
    assert old_scope is not None

    def BuildScope(builder):
        nonlocal scope
        scope = build_func(old_scope, builder)
        assert scope is not None

    oneflow_api.deprecated.LogicalRun(BuildScope)
    return scope


def MakeInitialScope(job_conf, device_tag, machine_device_ids, hierarchy, is_mirrored):
    scope = None

    def BuildInitialScope(builder):
        nonlocal scope
        session_id = session_ctx.GetDefaultSession().id
        scope = builder.BuildInitialScope(
            session_id, job_conf, device_tag, machine_device_ids, hierarchy, is_mirrored
        )

    oneflow_api.deprecated.LogicalRun(BuildInitialScope)
    return scope


def InitScopeStack():
    job_conf = job_conf_cfg.JobConfigProto()
    job_conf.mutable_predict_conf()
    job_conf.set_job_name("")
    scope = MakeInitialScope(job_conf, "cpu", ["0:0"], None, is_mirrored=False)
    oneflow_api.InitGlobalScopeStack(scope)


@contextmanager
def ScopeContext(scope):
    old_scope = oneflow_api.GetCurrentScope()
    oneflow_api.GlobalScopeStackPush(scope)
    try:
        yield
    finally:
        assert oneflow_api.GetCurrentScope() is scope
        oneflow_api.GlobalScopeStackPop()
        assert oneflow_api.GetCurrentScope() is old_scope
