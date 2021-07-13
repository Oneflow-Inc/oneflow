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

import oneflow._oneflow_internal


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

from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.core.job.job_conf_pb2 import JobConfigProto

from oneflow.compatible.single_client.python.framework import session_util
from oneflow.compatible.single_client.python.framework import session_context
from oneflow.compatible.single_client.python.framework import env_util

oneflow._oneflow_internal.DestroyEnv()
oneflow._oneflow_internal.SetIsMultiClient(False)
env_util.init_default_physical_env()
session_context.OpenDefaultSession(
    session_util.Session(oneflow._oneflow_internal.NewSessionId())
)

del env_util
del session_util
del session_context


import oneflow.compatible.single_client.python_gen.__export_symbols__

import oneflow.compatible.single_client.python.framework.c_api_util

# register ForeignCallback
from oneflow.compatible.single_client.python.framework import register_python_callback
from oneflow.compatible.single_client.python.framework import python_callback

oneflow._oneflow_internal.RegisterForeignCallbackOnlyOnce(
    python_callback.global_python_callback
)
del python_callback
del register_python_callback

# register Watcher
from oneflow.compatible.single_client.python.framework import watcher

oneflow._oneflow_internal.RegisterWatcherOnlyOnce(watcher._global_watcher)
del watcher

# register BoxingUtil
from oneflow.compatible.single_client.python.eager import boxing_util

oneflow._oneflow_internal.deprecated.RegisterBoxingUtilOnlyOnce(
    boxing_util._global_boxing_util
)
del boxing_util


from oneflow.compatible.single_client.python.framework import register_class_method_util

register_class_method_util.RegisterMethod4Class()
del register_class_method_util

INVALID_SPLIT_AXIS = oneflow._oneflow_internal.INVALID_SPLIT_AXIS

import atexit
from oneflow.compatible.single_client.python.framework.session_context import (
    TryCloseDefaultSession,
)

atexit.register(TryCloseDefaultSession)

del TryCloseDefaultSession
del atexit
del absolute_import
