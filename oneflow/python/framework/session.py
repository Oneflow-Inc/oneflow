from __future__ import absolute_import

import threading
from oneflow.core.job.job_set_pb2 import JobSet
from oneflow.core.job.job_set_pb2 import ConfigProto
import oneflow.core.job.job_pb2 as job_util
import oneflow.python.framework.runtime as runtime
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.runtime_context as runtime_ctx
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.job_set_util as job_set_util
from oneflow.python.framework.out_remote_blobs_result_box import OutRemoteBlobsResultBox
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('init')
def init(config_proto):
    if (isinstance(config_proto, config_util.ConfigProtoBuilder)):
        config_proto = config_proto.config_proto
    assert isinstance(config_proto, ConfigProto)
    config_util.TryCompleteDefaultConfigProto(config_proto)
    config_util.inited_config_proto = config_proto
    c_api_util.Init(config_proto)

@oneflow_export('Session')
class Session(object):
    def __init__(self, job_set = None):
        if job_set == None: job_set = job_set_util.get_default_job_set()
        assert isinstance(job_set, JobSet)
        self.job_set_ = job_set
        self.job_name2job_func_ = job_set_util.GetJobName2JobFunc(job_set)
        self.is_running_ = False
        self.cond_var_ = threading.Condition()
        self.running_job_cnt_ = 0
        runtime_ctx.AddJobInstancePreLaunchCallbacks(self._PreLaunchCallback)
        runtime_ctx.AddJobInstancePostFinishCallbacks(self._PostFinishCallback)

    def run(self, job_func, *arg):
        assert self.is_running_
        assert job_func.__name__ in self.job_name2job_func_
        return self.Run(job_func, *arg)

    def map(self, job_func, feed_data):
        assert self.is_running_
        assert job_func.__name__ in self.job_name2job_func_
        return self.Map(job_func, feed_data)

    def no_return_run(self, job_func, *arg):
        assert self.is_running_
        assert job_func.__name__ in self.job_name2job_func_
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
        return OutRemoteBlobsResultBox().SetResult(remote_blobs).Inited()

    def Map(self, job_func, feed_data):
        result_box = OutRemoteBlobsResultBox()
        for x in feed_data:
            if not (isinstance(x, list) or isinstance(x, tuple)): x = [x]
            remote_blobs = runtime.LaunchJob(job_func, *x)
            assert remote_blobs is not None
            result_box.AddResult(remote_blobs)
        result_box.Inited()
        return result_box

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

    def __enter__(self):
        assert self.is_running_ == False
        self.is_running_ = True
        self.runtime_env_ = runtime.GetMachineRuntimeEnv(self.job_set_)
        self.runtime_env_.__enter__()
        runtime_ctx.default_session = self
        return self

    def __exit__(self, *args):
        assert self.is_running_ == True
        self.runtime_env_.__exit__()

