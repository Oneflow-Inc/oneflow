from __future__ import absolute_import

import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.core.job.job_pb2 as job_util
import oneflow.python.framework.compiler as compiler
import oneflow.python.framework.runtime as runtime
import oneflow.python.framework.config_util as config_util

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

    def run(self, job_func, *arg):
        assert self.is_running_
        assert job_func.__name__ in self.job_name2job_func_
        return runtime.LaunchJob(job_func, *arg)
        
    def map(self, job_func, feed_dicts):
        assert self.is_running_
        assert job_func.__name__ in self.job_name2job_func_
        TODO()

    def sync(self):
        assert self.is_running_
        TODO()
    
    def __enter__(self):
        assert self.is_running_ == False
        self.is_running_ = True
        self.runtime_env_ = runtime.GetMachineRuntimeEnv(self.job_set_)
        self.runtime_env_.__enter__()
        return self

    def __exit__(self, *args):
        assert self.is_running_ == True
        self.runtime_env_.__exit__()
