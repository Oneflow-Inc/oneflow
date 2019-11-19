from __future__ import absolute_import

import threading
from oneflow.core.job.job_set_pb2 import ConfigProto
import oneflow.core.job.job_pb2 as job_util
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.framework.session_context import SessionStatus
import oneflow.python.framework.compiler as compiler
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.env_util as env_util
import oneflow.python.framework.job_instance as job_instance_util
from oneflow.python.framework.out_remote_blobs_status import OutRemoteBlobsStatus
from oneflow.python.oneflow_export import oneflow_export

class Session(object):
    def __init__(self):
        self.job_name2job_func_ = {}
        self.status_ = SessionStatus.OPEN
        self.cond_var_ = threading.Condition()
        self.running_job_cnt_ = 0
        self.inter_user_job_info_ = None

    @property
    def inter_user_job_info(self): return self.inter_user_job_info_

    @property
    def status(self): return self.status_

    def TryInit(self):
        if self.status_ is SessionStatus.OPEN: self.Init()
        return self
    
    def Init(self):
        assert self.status_ is SessionStatus.OPEN
        _TryInitEnv()
        c_api_util.InitGlobalSession(_GetConfigProto())
        for job_name, job_func in self.job_name2job_func_.items(): compiler.Compile(job_func)
        c_api_util.StartGlobalSession()
        self.inter_user_job_info_ = c_api_util.GetInterUserJobInfo()
        self.status_ = SessionStatus.RUNNING
        return self

    def TryClose(self):
        if self.status_ is SessionStatus.RUNNING: self.Close()

    def Close(self):
        assert self.status_ is SessionStatus.RUNNING
        c_api_util.StopGlobalSession()
        c_api_util.DestroyGlobalSession()
        self.Sync()
        self.status_ = SessionStatus.CLOSED

    def AddJob(self, job_func):
        assert self.status_ is SessionStatus.OPEN
        self.job_name2job_func_[job_func.__name__] = job_func

    def Sync(self):
        assert self.status_ is SessionStatus.RUNNING
        self.cond_var_.acquire()
        while self.running_job_cnt_ > 0:
            self.cond_var_.wait()
        assert self.running_job_cnt_ == 0
        self.cond_var_.release()

    def Run(self, job_func, *arg):
        assert self.status_ is SessionStatus.RUNNING
        remote_blobs = self.LaunchUserJob(job_func, *arg)
        if remote_blobs is None: return
        return OutRemoteBlobsStatus(self).SetResult(remote_blobs).Inited()
    
    def LaunchUserJob(self, job_func, *arg):
        assert self.status_ is SessionStatus.RUNNING
        job_name = job_func.__name__
        assert len(arg) == len(job_func.__oneflow_input_blob_defs__)
        for i in range(len(arg)):
            arg_blob = job_func.__oneflow_input_blob_defs__[i]
            arg_blob.CheckInputNdarray(arg[i])
            self.AsyncPush(arg_blob.op_name, _MakePushCallback(arg[i]))
        self.LaunchJob(job_instance_util.MakeUserJobInstance(job_name))
        return job_func.__oneflow_output_remote_blobs__

    def LaunchJob(self, job_instance):
        assert self.status_ is SessionStatus.RUNNING
        self._IncRunningJobCnt()
        job_instance.AddPostFinishCallback(lambda _: self._DecRunningJobCnt())
        c_api_util.LaunchJob(job_instance)

    def AsyncPush(self, op_name, push_data_cb):
        assert self.status_ is SessionStatus.RUNNING
        push_job_name = self.inter_user_job_info.input_or_var_op_name2push_job_name[op_name]
        self.LaunchJob(job_instance_util.MakePushJobInstance(push_job_name, op_name, push_data_cb))

    def AsyncPull(self, op_name, pull_data_cb):
        assert self.status_ is SessionStatus.RUNNING
        pull_job_name = self.inter_user_job_info.output_or_var_op_name2pull_job_name[op_name]
        self.LaunchJob(job_instance_util.MakePullJobInstance(pull_job_name, op_name, pull_data_cb))
    
    def _IncRunningJobCnt(self):
        assert self.status_ is SessionStatus.RUNNING
        self.cond_var_.acquire()
        self.running_job_cnt_ += 1
        self.cond_var_.release()

    def _DecRunningJobCnt(self):
        self.cond_var_.acquire()
        self.running_job_cnt_ -= 1
        self.cond_var_.notify()
        self.cond_var_.release()

@oneflow_export("clear_default_session")
def clear_default_session():
    session_ctx.TryCloseDefaultSession()
    session_ctx.OpenDefaultSession(Session())

def _MakePushCallback(ndarray):
    return lambda ofblob: ofblob.CopyFromNdarray(ndarray)

def _GetConfigProto():
    config_proto = config_util.default_config_proto
    if config_proto.resource.machine_num <= 0:
      config_proto.resource.machine_num = \
          len(env_util.default_env_proto.machine)
    return config_proto

def _TryInitEnv():
    if c_api_util.IsEnvInited(): return
    assert len(env_util.default_env_proto.machine) > 0
    env_util.CompleteEnvProto(env_util.default_env_proto)
    c_api_util.InitEnv(env_util.default_env_proto)
    env_util.env_proto_mutable = False

session_ctx.OpenDefaultSession(Session())
