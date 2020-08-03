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
from oneflow.python.version import __version__

from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.core.job.job_conf_pb2 import JobConfigProto
import oneflow.python.framework.session_util as session_util
del session_util

import oneflow.python.framework.register_python_callback

import oneflow.python.__export_symbols__

import atexit
import oneflow.python.framework.c_api_util
atexit.register(oneflow.python.framework.c_api_util.DestroyEnv)
atexit.register(oneflow.python.framework.session_context.TryCloseDefaultSession)
del atexit

del absolute_import
del oneflow
del python
