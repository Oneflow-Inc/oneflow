from __future__ import absolute_import

import threading
from oneflow.core.job.job_set_pb2 import ConfigProto
import oneflow.core.job.job_pb2 as job_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.job_instance as job_instance_util
from oneflow.python.framework.out_remote_blobs_status import OutRemoteBlobsStatus
from oneflow.python.oneflow_export import oneflow_export

class Session(object):
    def __init__(self):
        self.job_name2job_func_ = {}
        self.is_running_ = False
        self.cond_var_ = threading.Condition()
        self.running_job_cnt_ = 0
        self.inter_user_job_info_ = None

    @property
    def inter_user_job_info(self): return self.inter_user_job_info_

    @property
    def is_running(self): return self.is_running_

    def Init(self):
        assert self.is_running_ == False
        TryInitEnvironment()
        for job_name, job_func in self.job_name2job_func_.items(): compiler.Compile(job_func)
        c_api_util.InitGlobalOneflow()
        self.inter_user_job_info_ = c_api_util.GetInterUserJobInfo()
        self.is_running_ = True

    def Destory(self):
        c_api_util.DestroyGlobalOneflow()

    def AddJob(self, job_func):
        self.job_name2job_func_[job_func.__name__] = job_func

    def Sync(self):
        assert self.is_running_
        self.cond_var_.acquire()
        while self.running_job_cnt_ > 0:
            self.cond_var_.wait()
        assert self.running_job_cnt_ == 0
        self.cond_var_.release()

    def Run(self, job_func, *arg):
        assert self.is_running_
        remote_blobs = self.LaunchUserJob(job_func, *arg)
        if remote_blobs is None: return
        return OutRemoteBlobsStatus(self).SetResult(remote_blobs).Inited()
    
    def LaunchUserJob(self, job_func, *arg):
        assert self.is_running_
        job_name = job_func.__name__
        assert len(arg) == len(job_func.__oneflow_input_blob_defs__)
        for i in range(len(arg)):
            arg_blob = job_func.__oneflow_input_blob_defs__[i]
            arg_blob.CheckInputNdarray(arg[i])
            self.AsyncPush(arg_blob.op_name, _MakePushCallback(arg[i]))
            self.LaunchJob(job_instance_util.MakeUserJobInstance(job_name))
        return job_func.__oneflow_output_remote_blobs__

    def LaunchJob(self, job_instance):
        assert self.is_running_
        self._IncRunningJobCnt()
        job_instance.AddPostFinishCallback(lambda _: self._DecRunningJobCnt)
        c_api_util.LaunchJob(job_instance)

    def AsyncPush(self, op_name, push_data_cb):
        assert self.is_running_
        push_job_name = self.inter_user_job_info.input_or_var_op_name2push_job_name[op_name]
        self.LaunchJob(job_instance_util.MakePushJobInstance(push_job_name, op_name, push_data_cb))

    def AsyncPull(self, op_name, pull_data_cb):
        assert self.is_running_
        pull_job_name = self.inter_user_job_info.output_or_var_op_name2pull_job_name[op_name]
        self.LaunchJob(job_instance_util.MakePullJobInstance(pull_job_name, op_name, pull_data_cb))
    
    def _IncRunningJobCnt(self):
        assert self.is_running_
        self.cond_var_.acquire()
        self.running_job_cnt_ += 1
        self.cond_var_.release()

    def _DecRunningJobCnt(self):
        assert self.is_running_
        self.cond_var_.acquire()
        self.running_job_cnt_ -= 1
        self.cond_var_.notify()
        self.cond_var_.release()

def _MakePushCallback(ndarray):
    return lambda ofblob: ofblob.CopyFromNdarrayOrNestedNdarrayList(ndarray)

def TryInitEnvironment():
    if c_api_util.IsEnvironmentInited() == False:
        c_api_util.InitEnvironment(config_util.default_config_proto)
        config_util.config_proto_mutable = False

session_ctx.ResetDefaultSession(Session())
