from __future__ import absolute_import

import threading
import functools
from oneflow.core.job.job_set_pb2 import ConfigProto
import oneflow.core.job.job_pb2 as job_util
import oneflow.python.framework.runtime as runtime
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.runtime_context as runtime_ctx
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.job_set_util as job_set_util
from oneflow.python.framework.out_remote_blobs_status import OutRemoteBlobsStatus
from oneflow.python.oneflow_export import oneflow_export

def try_init_default_session(func):
    @functools.wraps(func)
    def Func(*args):
        assert GetDefaultSession() != None
        return func(*args)
    return Func

class Session(object):
    def __init__(self):
        self.is_running_ = False
        self.cond_var_ = threading.Condition()
        self.running_job_cnt_ = 0
        runtime_ctx.AddJobInstancePreLaunchCallbacks(self._PreLaunchCallback)
        runtime_ctx.AddJobInstancePostFinishCallbacks(self._PostFinishCallback)
        assert c_api_util.IsEnvironmentInited() == False
        c_api_util.InitEnvironment(config_util.default_config_proto)
        config_util.config_proto_mutable = False
        job_set_util.compile_all_job()
        self.is_running_ = True
        self.runtime_env_ = runtime.GetMachineRuntimeEnv()

    def run(self, job_func, *arg):
        assert self.is_running_
        return self.Run(job_func, *arg)

    def map(self, job_func, feed_data):
        assert self.is_running_
        return self.Map(job_func, feed_data)

    def no_return_run(self, job_func, *arg):
        assert self.is_running_
        return self.NoReturnRun(job_func, *arg)

    def sync(self):
        assert self.is_running_
        self.cond_var_.acquire()
        while self.running_job_cnt_ > 0:
            self.cond_var_.wait()
        assert self.running_job_cnt_ == 0
        self.cond_var_.release()

    def Run(self, job_func, *arg):
        remote_blobs = runtime.LaunchJob(job_func, *arg)
        if remote_blobs is None: return
        return OutRemoteBlobsStatus().SetResult(remote_blobs).Inited()

    def Map(self, job_func, feed_data):
        status = OutRemoteBlobsStatus()
        for x in feed_data:
            if not (isinstance(x, list) or isinstance(x, tuple)): x = [x]
            remote_blobs = runtime.LaunchJob(job_func, *x)
            assert remote_blobs is not None
            status.AddResult(remote_blobs)
        status.Inited()
        return status

    def NoReturnRun(self, job_func, *arg):
        runtime.LaunchJob(job_func, *arg)

    def _PreLaunchCallback(self, job_instance):
        self.cond_var_.acquire()
        self.running_job_cnt_ += 1
        self.cond_var_.release()

    def _PostFinishCallback(self, job_instance):
        self.cond_var_.acquire()
        self.running_job_cnt_ -= 1
        self.cond_var_.notify()
        self.cond_var_.release()

    def InitMasterRuntimeEnv(self):
        assert self.is_running_ == False
        self.is_running_ = True
        return self

def GetDefaultSession():
    global _default_session
    if _default_session == None:
        _default_session = Session()
    return _default_session

_default_session = None
