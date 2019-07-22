from __future__ import absolute_import

import threading
import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.core.job.job_pb2 as job_util
import oneflow.python.framework.compiler as compiler
import oneflow.python.framework.runtime as runtime
import oneflow.python.framework.runtime_context as runtime_ctx
import oneflow.python.framework.config_util as config_util
from oneflow.python.framework.out_remote_blobs_result_box import OutRemoteBlobsResultBox

class Session(object):
    def __init__(self, job_funcs, config_proto):
        if (isinstance(config_proto, config_util.ConfigProtoBuilder)):
            config_proto = config_proto.config_proto
        assert isinstance(config_proto, job_set_util.ConfigProto)
        self.job_set_ = job_set_util.JobSet()
        self.job_set_.config.CopyFrom(config_proto)
        compiler.Compile(self.job_set_, job_funcs)
        self.job_name2job_func_ = {}
        for job_func in job_funcs: self.job_name2job_func_[job_func.__name__] = job_func
        self.is_running_ = False
        self.cond_var_ = threading.Condition()
        self.running_job_cnt_ = 0
        runtime_ctx.AddJobInstancePreLaunchCallbacks(self.pre_launch_callback)
        runtime_ctx.AddJobInstancePostFinishCallbacks(self.post_finish_callback)

    def run(self, job_func, *arg):
        assert self.is_running_
        assert job_func.__name__ in self.job_name2job_func_
        remote_blobs = runtime.LaunchJob(job_func, *arg)
        if remote_blobs is None: return
        return OutRemoteBlobsResultBox().SetResult(remote_blobs).Inited()

    def map(self, job_func, feed_data):
        assert self.is_running_
        assert job_func.__name__ in self.job_name2job_func_
        result_box = OutRemoteBlobsResultBox()
        for x in feed_data:
            if not (isinstance(x, list) or isinstance(x, tuple)): x = [x]
            remote_blobs = runtime.LaunchJob(job_func, *x)
            assert remote_blobs is not None
            result_box.AddResult(remote_blobs)
        result_box.Inited()
        return result_box

    def no_return_run(self, job_func, *arg):
        assert self.is_running_
        assert job_func.__name__ in self.job_name2job_func_
        runtime.LaunchJob(job_func, *arg)

    def sync(self):
        assert self.is_running_
        self.cond_var_.acquire()
        while self.running_job_cnt_ > 0:
            self.cond_var_.wait()
        assert self.running_job_cnt_ == 0
        self.cond_var_.release()

    def pre_launch_callback(self, job_instance):
        self.cond_var_.acquire()
        self.running_job_cnt_ += 1
        self.cond_var_.release()

    def post_finish_callback(self, job_instance):
        self.cond_var_.acquire()
        self.running_job_cnt_ -= 1
        self.cond_var_.notify()
        self.cond_var_.release()

    def __enter__(self):
        assert self.is_running_ == False
        self.is_running_ = True
        self.runtime_env_ = runtime.GetMachineRuntimeEnv(self.job_set_)
        self.runtime_env_.__enter__()
        return self

    def __exit__(self, *args):
        assert self.is_running_ == True
        self.runtime_env_.__exit__()

