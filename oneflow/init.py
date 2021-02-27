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


def import_secondary_module(name, path):
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.ExtensionFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def import_oneflow_internal2():
    import oneflow
    import os
    from os.path import dirname
    import imp

    fp, pathname, description = imp.find_module(
        "_oneflow_internal", [dirname(__file__)]
    )
    assert os.path.isfile(pathname)
    return import_secondary_module("oneflow_api", pathname)


oneflow_api = import_oneflow_internal2()

# define dtype at the begining of oneflow init

locals()["dtype"] = oneflow_api.dtype
locals()["char"] = oneflow_api.char
locals()["float16"] = oneflow_api.float16
locals()["float32"] = oneflow_api.float32
locals()["float"] = oneflow_api.float
locals()["double"] = oneflow_api.double
locals()["float64"] = oneflow_api.float64
locals()["int8"] = oneflow_api.int8
locals()["int32"] = oneflow_api.int32
locals()["int64"] = oneflow_api.int64
locals()["uint8"] = oneflow_api.uint8
locals()["record"] = oneflow_api.record
locals()["tensor_buffer"] = oneflow_api.tensor_buffer

del import_secondary_module
del import_oneflow_internal2

from oneflow.python.version import __version__

from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.core.job.job_conf_pb2 import JobConfigProto
import oneflow.python.framework.session_util as session_util

del session_util

import oneflow.python.framework.register_python_callback

import oneflow.python_gen.__export_symbols__

import atexit
import oneflow.python.framework.c_api_util
import oneflow.python.framework.python_interpreter_util
import oneflow.python.framework.register_class_method_util as register_class_method_util
import oneflow_api

INVALID_SPLIT_AXIS = oneflow_api.INVALID_SPLIT_AXIS

register_class_method_util.RegisterMethod4Class()

atexit.register(oneflow_api.DestroyEnv)
atexit.register(oneflow.python.framework.session_context.TryCloseDefaultSession)
atexit.register(oneflow.python.framework.python_interpreter_util.SetShuttingDown)
del atexit

import sys

__original_exit__ = sys.exit


def custom_exit(returncode):
    if returncode != 0:
        oneflow_api.MasterSendAbort()
    __original_exit__(returncode)


sys.exit = custom_exit

del custom_exit
del sys
del absolute_import
del oneflow
del python
del python_gen
