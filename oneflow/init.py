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
# __init__.py, rename to avoid being added to PYTHONPATH
from __future__ import absolute_import

import oneflow._oneflow_internal

oneflow._oneflow_internal.CheckAndClearRegistryFlag()

Size = oneflow._oneflow_internal.Size
device = oneflow._oneflow_internal.device
placement = oneflow._oneflow_internal.placement
no_grad = oneflow._oneflow_internal.autograd.no_grad

# define dtype at the begining of oneflow init

locals()["dtype"] = oneflow._oneflow_internal.dtype
locals()["char"] = oneflow._oneflow_internal.char
locals()["float16"] = oneflow._oneflow_internal.float16
locals()["half"] = oneflow._oneflow_internal.float16
locals()["float32"] = oneflow._oneflow_internal.float32
locals()["float"] = oneflow._oneflow_internal.float
locals()["double"] = oneflow._oneflow_internal.double
locals()["float64"] = oneflow._oneflow_internal.float64
locals()["int8"] = oneflow._oneflow_internal.int8
locals()["int"] = oneflow._oneflow_internal.int32
locals()["int32"] = oneflow._oneflow_internal.int32
locals()["int64"] = oneflow._oneflow_internal.int64
locals()["long"] = oneflow._oneflow_internal.int64
locals()["uint8"] = oneflow._oneflow_internal.uint8
locals()["record"] = oneflow._oneflow_internal.record
locals()["tensor_buffer"] = oneflow._oneflow_internal.tensor_buffer

from oneflow.python.version import __version__

from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.core.job.job_conf_pb2 import JobConfigProto


import oneflow.python.framework.register_python_callback

import oneflow.python_gen.__export_symbols__

import atexit
import oneflow.python.framework.c_api_util
import oneflow.python.framework.register_class_method_util as register_class_method_util

INVALID_SPLIT_AXIS = oneflow._oneflow_internal.INVALID_SPLIT_AXIS

register_class_method_util.RegisterMethod4Class()
oneflow._oneflow_internal.RegisterGILForeignLockHelper()

import oneflow.python.framework.env_util as env_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.scope_util as scope_util
from oneflow.python.framework.session_util import Session
from oneflow.python.framework.multi_client_session import MultiClientSession


if not env_util.HasAllMultiClientEnvVars():
    env_util.SetDefaultMultiClientEnvVars()
oneflow._oneflow_internal.SetIsMultiClient(True)
env_util.api_env_init()
session_ctx.OpenDefaultSession(
    MultiClientSession(oneflow._oneflow_internal.NewSessionId())
)
scope_util.InitScopeStack()
oneflow._oneflow_internal.EnableEagerEnvironment(True)

del env_util

# NOTE(chengcheng): register ForeignCallback and Watcher used by nn.Graph train job compelete
from oneflow.python.framework import register_python_callback
from oneflow.python.framework import python_callback

oneflow._oneflow_internal.RegisterGlobalForeignCallback(
    python_callback.global_python_callback
)
del python_callback
del register_python_callback

from oneflow.python.framework import watcher

oneflow._oneflow_internal.RegisterGlobalWatcher(watcher._global_watcher)
del watcher


def _SyncOnMasterFn():
    import oneflow

    def Sync():
        if not oneflow._oneflow_internal.IsEnvInited():
            return
        if oneflow.python.framework.distribute.is_multi_client():
            oneflow._oneflow_internal.eager.multi_client.Sync()
        elif oneflow.python.framework.distribute.get_rank() == 0:
            oneflow._oneflow_internal.eager.single_client.Sync()

    return Sync


atexit.register(oneflow._oneflow_internal.SetShuttingDown)
atexit.register(oneflow._oneflow_internal.DestroyEnv)
atexit.register(oneflow.python.framework.session_context.TryCloseDefaultSession)
# Global<ResourceDesc, ForSession>::Get(), used by vm in background thread,
# will be set to nullptr by TryCloseDefaultSession,
# so sync vm in advance to avoid data race
atexit.register(_SyncOnMasterFn)

del atexit

del absolute_import
del oneflow

import oneflow.python.framework.docstr as docstr
from oneflow.python.framework.docstr.utils import register_docstr

register_docstr()
del register_docstr
del docstr
