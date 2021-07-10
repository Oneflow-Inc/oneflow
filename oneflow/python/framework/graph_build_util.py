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

import inspect
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.runtime_mode as runtime_mode
import oneflow.python.framework.scope_util as scope_util
import oneflow._oneflow_internal

lazy_mode = oneflow._oneflow_internal.lazy_mode


@contextmanager
def graph_build_context(config):
    with lazy_mode.gard(True):
        with JobBuildAndInferCtx(config):
            yield
    # with runtime_mode.ModeScope(runtime_mode.GLOBAL_MODE):
    # with scope_util.ScopeContext(scope):
    #     yield


class JobBuildAndInferCtx(object):
    def __init__(self, config):
        self._job_conf = config.proto

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
