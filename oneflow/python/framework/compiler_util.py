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

import inspect
from typing import Callable, Union

import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.typing_util as oft_util
from oneflow.python.framework.function_util import (
    FunctionConfig as ExecutionConfig,
    _CloneFunctionDesc,
)
from oneflow.python.framework.env_util import api_enable_eager_execution
from oneflow.python.oneflow_export import oneflow_export


class GraphFunction(object):
    def __init__(self, func=None, function_config=ExecutionConfig()):
        assert func is not None
        self._func = func
        if not hasattr(self._func, "__oneflow_function_signature__"):
            self._func.__oneflow_function_signature__ = inspect.signature(func)
        oft_util.CheckGlobalFunctionAnnotation(
            self._func.__oneflow_function_signature__
        )
        self._sess = session_ctx.GetDefaultSession()
        self._sess.AddJob(_CloneFunctionDesc(function_config.function_desc, self._func))

    def __call__(self, *args, **kwargs):
        return self._sess.TryInit().LazyRun(self._func, *args, **kwargs)


@oneflow_export("compiler.trace")
def compiler_trace(
    func=None, *, execution_config: ExecutionConfig = None, type: str = "predict",
) -> Union[Callable[[Callable], GraphFunction], GraphFunction]:
    api_enable_eager_execution(False)
    if execution_config is None:
        execution_config = ExecutionConfig()
    if type == "train":
        execution_config.function_desc.job_config_proto.mutable_train_conf()
    else:
        execution_config.function_desc.job_config_proto.mutable_predict_conf()
    # TODO(): rm type
    assert type in ["train", "predict"]
    if func is None:
        # Using compile_trace as decorator
        return _graph_function_deco(execution_config)
    else:
        # Using compile_trace as function
        return _graph_function_deco(execution_config)(func)


def _graph_function_deco(function_config=ExecutionConfig()):
    assert isinstance(function_config, ExecutionConfig)

    def _decorator(func):
        graph_func = GraphFunction(func, function_config)
        return graph_func

    return _decorator
