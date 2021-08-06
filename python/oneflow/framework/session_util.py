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
import inspect
import threading
import traceback
from contextlib import contextmanager
from typing import Callable

from google.protobuf import text_format

import oneflow
import oneflow._oneflow_internal
import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.eager.op_executor as op_executor
import oneflow.framework.c_api_util as c_api_util
import oneflow.framework.check_point_v2 as check_point_v2
import oneflow.framework.compiler as compiler
import oneflow.framework.config_util as config_util
import oneflow.framework.env_util as env_util
import oneflow.framework.hob as hob
import oneflow.framework.job_instance as job_instance_util
import oneflow.framework.module as module_util
import oneflow.framework.push_util as push_util
import oneflow.framework.session_context as session_ctx
import oneflow.framework.typing_util as oft_util
import oneflow.support.enable_if as enable_if
from oneflow import oneflow_deprecate
from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.experimental import interface_op_read_and_write
from oneflow.framework.check_point import SnapshotManager
from oneflow.framework.function_desc import FunctionDesc
from oneflow.framework.pull_util import EagerFutureRemoteBlobs, LazyFutureRemoteBlobs
from oneflow.framework.session_context import SessionStatus


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
    elif not reuse:
        assert module_name not in sess.existed_module_names_, (
            "duplicated module_name `%s' in global_function `%s'"
            % (module_name, job_name)
        )
    else:
        pass
    sess.existed_module_names_.add(module_name)
    return module_name2module[module_name]


def api_eager_execution_enabled() -> bool:
    """Get current setting of the job, if enable eager execution mode ,then return True

    Returns:
        bool: [description]
    """
    return oneflow._oneflow_internal.EagerExecutionEnabled()


def api_clear_default_session() -> None:
    """Clear the default session. All compiled OneFlow functions will be deleted.
    """
    func = enable_if.unique([clear_default_session])
    return func()


@enable_if.condition(hob.in_normal_mode)
def clear_default_session():
    is_multi_client = oneflow._oneflow_internal.IsMultiClient()
    if not is_multi_client:
        session_ctx.TryCloseDefaultSession()
        session_ctx.OpenDefaultSession(
            Session(oneflow._oneflow_internal.NewSessionId())
        )


def api_sync_default_session() -> None:
    """Synchronize the default session. Block until every synchronous OneFlow function and its callback finishes running.
    """
    func = enable_if.unique([sync_default_session])
    return func()


@enable_if.condition(hob.in_normal_mode)
def sync_default_session() -> None:
    session_ctx.GetDefaultSession().Sync()


def _TryCompleteConfigProto(config_proto):
    if config_proto.resource.machine_num == 0:
        config_proto.resource.machine_num = oneflow._oneflow_internal.GetNodeSize()


def _GetDefaultConfigProto():
    config_proto = job_set_util.ConfigProto()
    config_proto.resource.machine_num = 0
    if oneflow._oneflow_internal.flags.with_cuda():
        config_proto.resource.gpu_device_num = 1
    else:
        config_proto.resource.cpu_device_num = 1
        config_proto.resource.gpu_device_num = 0
    config_proto.session_id = session_ctx.GetDefaultSession().id
    return config_proto


def TmpInitEagerGlobalSession():
    config_pb = _GetDefaultConfigProto()
    config_proto_str = text_format.MessageToString(config_pb)
    oneflow._oneflow_internal.InitEagerGlobalSession(config_proto_str)
