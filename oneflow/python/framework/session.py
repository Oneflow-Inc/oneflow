from __future__ import absolute_import

import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.core.job.job_conf_pb2 as job_conf_util
import oneflow.python.framework.compiler as compiler
import oneflow.python.framework.runtime as runtime
import oneflow.python.framework.config_util as config_util

class Session(object):
    def __init__(self):
        self.job_set_ = job_set_util.JobSet()
        self.running_ = False
        self.job_func_ = []
        self.job_name2job_conf_ = {}
        self.job_name2job_func_ = {}
        self.check_unique_job_func_name_ = set()

    # compile api
    def config(self):
        return config_util.JobSetConfiger(self.job_set_)

    # compile api
    def add_job(self, job_func):
        assert job_func.__name__ not in self.check_unique_job_func_name_
        self.job_func_.append(job_func)
        self.job_name2job_func_[job_func.__name__] = job_func
        job_conf = self.job_set_.job_conf.add()
        job_conf.job_name = job_func.__name__
        self.job_name2job_conf_[job_func.__name__] = job_conf
        return self.config_job(job_func.__name__)
        
    # compile api
    def config_job(self, job_name):
        assert job_name in self.job_name2job_conf_
        return config_util.JobConfiger(self.job_name2job_conf_[job_name])
    
    # compile api
    def compile_only(self):
        compiler.Compile(self.job_set_, lambda job_name : self.job_name2job_func_[job_name])
        return self.job_set_

    # runtime api
    def run(self, job_func, *arg):
        assert job_func.__name__ in self.job_name2job_conf_
        return runtime.LaunchJob(job_func.__name__, *arg)
        
    # runtime api
    def map(self, job_func, feed_dicts):
        assert job_func.__name__ in self.job_name2job_conf_
        TODO()

    # runtime api
    def sync(self):
        TODO()
        
    def __enter__(self):
        assert self.running_ == False
        self.running_ = True
        job_set = self.compile_only()
        self.runtime_env_ = runtime.GetMachineRuntimeEnv(job_set)
        self.runtime_env_.__enter__()

    def __exit__(self, *args):
        assert self.running_ == True
        self.runtime_env_.__leave__()
