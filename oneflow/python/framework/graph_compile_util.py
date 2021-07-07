
from __future__ import absolute_import

from contextlib import contextmanager

import inspect
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.runtime_mode as runtime_mode
import oneflow.python.framework.scope_util as scope_util
import oneflow._oneflow_internal


@contextmanager
def graph_build_context(config):
    # TODO(xuxiaoyu): set to lazy mode
    with JobBuildAndInferCtx(config):
        with runtime_mode.ModeScope(runtime_mode.GLOBAL_MODE):
            yield
            # with scope_util.ScopeContext(scope):
            #     yield

class JobBuildAndInferCtx(object):
    def __init__(self, config):
        self._job_conf = config.proto

    def __enter__(self):
        print("Graph ", self._job_conf.job_name() , " enter ctx.")
        c_api_util.JobBuildAndInferCtx_Open(self._job_conf.job_name())
        c_api_util.CurJobBuildAndInferCtx_SetJobConf(self._job_conf)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print("Graph ", self._job_conf.job_name() , " exit ctx.")
            # TODO(xuxiaoyu): open job optimization pass
            # oneflow._oneflow_internal.CurJobBuildAndInferCtx_Complete()
            oneflow._oneflow_internal.JobBuildAndInferCtx_Close()
            return True
        else:
            print("Graph ", self._job_conf.job_name() ," exit:", exc_type, exc_val, exc_tb)
            return False