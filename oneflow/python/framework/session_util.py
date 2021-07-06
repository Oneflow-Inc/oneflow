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

import oneflow
import oneflow._oneflow_internal
import oneflow.python.framework.hob as hob
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.framework.module as module_util

from oneflow.python.framework.session import _GetDefaultConfigProto
from oneflow.python.framework.single_client_session import SingleClientSession
from oneflow.python.framework.multi_client_session import MultiClientSession
from oneflow.python.oneflow_export import oneflow_export
from typing import Callable
from google.protobuf import text_format


@oneflow_export("find_or_create_module")
def api_find_or_create_module(
    module_name: str, create: Callable[[], None], reuse: bool = False
):
    func = enable_if.unique([find_or_create_module])
    return func(module_name, create, reuse)


@enable_if.condition(hob.in_global_mode)
def find_or_create_module(module_name, create, reuse=False):
    assert callable(create)
    sess = session_ctx.GetDefaultSession()
    job_name = oneflow.current_global_function_desc().job_config_proto.job_name()
    if job_name not in sess.job_name2module_name2module_:
        sess.job_name2module_name2module_[job_name] = {}
    module_name2module = sess.job_name2module_name2module_[job_name]
    if module_name not in module_name2module:
        module = create()
        assert isinstance(module, module_util.Module)
        module_name2module[module_name] = module
    else:
        if not reuse:
            assert module_name not in sess.existed_module_names_, (
                "duplicated module_name `%s' in global_function `%s'"
                % (module_name, job_name)
            )
        else:
            # do nothing
            pass
    sess.existed_module_names_.add(module_name)
    return module_name2module[module_name]


@oneflow_export("eager_execution_enabled")
def api_eager_execution_enabled() -> bool:
    """Get current setting of the job, if enable eager execution mode ,then return True

    Returns:
        bool: [description]
    """
    return oneflow._oneflow_internal.EagerExecutionEnabled()


@oneflow_export("clear_default_session")
def api_clear_default_session() -> None:
    r"""Clear the default session. All compiled OneFlow functions will be deleted.
    """
    func = enable_if.unique([clear_default_session])
    return func()


@enable_if.condition(hob.in_normal_mode)
def clear_default_session():
    session_ctx.TryCloseDefaultSession()
    session_ctx.OpenDefaultSession(_create_session())


@oneflow_export("sync_default_session")
def api_sync_default_session() -> None:
    r"""Synchronize the default session. Block until every synchronous OneFlow function and its callback finishes running.
    """
    func = enable_if.unique([sync_default_session])
    return func()


@enable_if.condition(hob.in_normal_mode)
def sync_default_session() -> None:
    sess = session_ctx.GetDefaultSession()
    if isinstance(sess, SingleClientSession):
        sess.Sync()
    else:
        raise TypeError


@oneflow_export("InitEagerGlobalSession")
def TmpInitEagerGlobalSession():
    config_pb = _GetDefaultConfigProto()
    config_proto_str = text_format.MessageToString(config_pb)
    oneflow._oneflow_internal.InitEagerGlobalSession(config_proto_str)


def _create_session():
    if oneflow.distributed.is_multi_client():
        return MultiClientSession(oneflow._oneflow_internal.NewSessionId())
    else:
        return SingleClientSession(oneflow._oneflow_internal.NewSessionId())


session_ctx.OpenDefaultSession(_create_session())
