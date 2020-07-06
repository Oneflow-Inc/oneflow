from __future__ import absolute_import

import threading
from oneflow.core.job.job_set_pb2 import ConfigProto
import oneflow.core.vm.instruction_pb2 as instr_util
import oneflow.core.eager.eager_symbol_pb2 as eager_symbol_util
import oneflow.core.job.job_pb2 as job_util
import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compiler as compiler
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.env_util as env_util
import oneflow.python.framework.hob as hob
import oneflow.python.framework.job_instance as job_instance_util
import oneflow.python.framework.push_util as push_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.eager.object_cache as object_cache
import oneflow.python.eager.vm_util as vm_util
from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.python.framework.function_desc import FunctionDesc
from oneflow.python.framework.pull_util import FutureRemoteBlobs
from oneflow.python.framework.session_context import SessionStatus
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.function_desc import FunctionDesc
from oneflow.python.eager.blob_register import BlobRegister
import oneflow.python.eager.blob_register as blob_register_util
from contextlib import contextmanager

import oneflow


class Session(object):
    def __init__(self):
        self.job_name2function_desc_ = {}
        self.status_ = SessionStatus.OPEN
        self.cond_var_ = threading.Condition()
        self.running_job_cnt_ = 0
        self.inter_user_job_info_ = None
        self.uuid2watch_handler_ = {}
        self.config_proto_ = None
        self.placement_scope_stack_ = []
        self.is_mirrored_strategy_enabled_stack_ = []
        self.function_flag_name2default_val_ = {}
        self.job_name2var_name2var_blob_ = {}
        self.var_name2var_blob_ = {}
        self.job_name2name_scope_stack_ = {}
        self.job_name2current_scope_ = {}
        self.eager_global_function_desc_stack_ = []
        self._UpdateFunctionFlagName2DefaultVal()
        self.instruction_list_ = instr_util.InstructionListProto()
        self.eager_symbol_list_ = eager_symbol_util.EagerSymbolList()
        self.backward_blob_register_ = BlobRegister()

    @property
    def status(self):
        return self.status_

    @property
    def is_running(self):
        return self.status_ is SessionStatus.RUNNING

    @property
    def config_proto(self):
        if self.config_proto_ is None:
            self.config_proto_ = _GetDefaultConfigProto()
        return self.config_proto_

    @property
    def uuid2watch_handler(self):
        return self.uuid2watch_handler_

    @property
    def placement_scope_stack(self):
        return self.placement_scope_stack_

    @property
    def is_mirrored_strategy_enabled_stack(self):
        return self.is_mirrored_strategy_enabled_stack_

    @property
    def function_flag_name2default_val(self):
        return self.function_flag_name2default_val_

    @property
    def inter_user_job_info(self):
        return self.inter_user_job_info_

    @property
    def job_name2name_scope_stack(self):
        return self.job_name2name_scope_stack_

    @property
    def instruction_list(self):
        return self.instruction_list_

    @property
    def eager_symbol_list(self):
        return self.eager_symbol_list_

    @property
    def backward_blob_register(self):
        return self.backward_blob_register_

    def MakeScope(self, build_func):
        scope = None
        old_scope = oneflow.scope.current_scope()
        assert old_scope is not None

        def BuildScope(builder):
            nonlocal scope
            scope = build_func(old_scope, builder)
            assert scope is not None

        vm_util.LogicalRun(BuildScope)
        return scope

    @contextmanager
    def NewCurrentScope(self, scope):
        job_name = scope.job_desc_symbol.data.job_name
        old_scope = self.GetCurrentScope(job_name)
        assert scope.parent_scope_symbol is old_scope
        self.job_name2current_scope_[job_name] = scope
        try:
            yield
        finally:
            assert self.GetCurrentScope(job_name) is scope
            self.job_name2current_scope_[job_name] = old_scope

    def InitNoneScope(self, job_name):
        if job_name not in self.job_name2current_scope_:
            assert isinstance(job_name, str)
            self.job_name2current_scope_[job_name] = None
        assert self.job_name2current_scope_[job_name] is None, "job_name: %s" % job_name

    def GetCurrentScope(self, job_name):
        assert job_name in self.job_name2current_scope_, "job_name: %s" % job_name
        return self.job_name2current_scope_[job_name]

    def GetLazyFunctionDesc(self, job_name):
        if job_name in self.job_name2function_desc_:
            return self.job_name2function_desc_[job_name]
        return None

    def AnyGlobalFunctionDefined(self):
        return len(self.job_name2function_desc_) > 0

    def GetJobConfigProto(self, job_name):
        return self.job_name2function_desc_[job_name].job_config_proto

    def GetFunctionDesc(self, job_name):
        return self.job_name2function_desc_[job_name]

    def _UpdateFunctionFlagName2DefaultVal(self):
        items = c_api_util.GetFunctionConfigDef().flag_name2flag_def.items()
        self.function_flag_name2default_val_ = {k: v.default_val for k, v in items}

    def TryInit(self):
        if self.status_ is SessionStatus.OPEN:
            self.Init()
        return self

    def Init(self):
        assert self.status_ is SessionStatus.OPEN
        self.status_ = SessionStatus.RUNNING
        if not c_api_util.IsEnvInited():
            oneflow.env.init()
        _TryCompleteConfigProto(self.config_proto)
        c_api_util.InitGlobalSession(self.config_proto)
        if not c_api_util.EagerExecutionEnabled():
            for job_name, func_desc in self.job_name2function_desc_.items():
                compiler.Compile(self, func_desc, self.config_proto)
            self.job_name2var_name2var_blob_ = dict()
            assert len(self.job_name2function_desc_.items()) > 0
            c_api_util.StartGlobalSession()
            self.inter_user_job_info_ = c_api_util.GetInterUserJobInfo()
        return self

    def TryClose(self):
        if self.status_ is SessionStatus.RUNNING:
            self.Close()

    def Close(self):
        assert self.status_ is SessionStatus.RUNNING
        self.Sync()
        assert len(self.job_name2var_name2var_blob_) == 0
        del self.var_name2var_blob_
        c_api_util.StopGlobalSession()
        c_api_util.DestroyGlobalSession()
        self.status_ = SessionStatus.CLOSED

    def AddJob(self, function_desc):
        assert self.status_ is SessionStatus.OPEN
        assert isinstance(function_desc, FunctionDesc)
        self.job_name2function_desc_[function_desc.job_func.__name__] = function_desc

    def Sync(self):
        assert self.status_ is SessionStatus.RUNNING
        self.cond_var_.acquire()
        while self.running_job_cnt_ > 0:
            self.cond_var_.wait()
        assert self.running_job_cnt_ == 0
        self.cond_var_.release()

    def LazyRun(self, job_func, *arg):
        assert self.status_ is SessionStatus.RUNNING
        remote_blobs = self.LaunchUserJob(job_func, *arg)
        if remote_blobs is None:
            return
        return FutureRemoteBlobs(self).SetResult(remote_blobs).Inited()

    def EagerRun(self, function_desc, *arg):
        with self._EagerGlobalFunctionDescScope(function_desc):
            return compiler.EagerRun(self, function_desc, self.config_proto, arg)

    def LaunchUserJob(self, job_func, *arg):
        assert self.status_ is SessionStatus.RUNNING
        job_name = job_func.__name__
        push_util.AsyncPush(self, job_func, *arg)
        self.LaunchJob(job_instance_util.MakeUserJobInstance(job_name))
        return job_func.__oneflow_output_remote_blobs__

    def LaunchJob(self, job_instance):
        assert self.status_ is SessionStatus.RUNNING
        self._IncRunningJobCnt()
        job_instance.AddPostFinishCallback(lambda _: self._DecRunningJobCnt())
        c_api_util.LaunchJob(job_instance)

    def AsyncPush(self, op_name, push_data_cb):
        assert self.status_ is SessionStatus.RUNNING
        push_job_name = self.inter_user_job_info.input_or_var_op_name2push_job_name[
            op_name
        ]
        self.LaunchJob(
            job_instance_util.MakePushJobInstance(push_job_name, op_name, push_data_cb)
        )

    def AsyncPull(self, op_name, pull_data_cb):
        assert self.status_ is SessionStatus.RUNNING
        pull_job_name = self.inter_user_job_info.output_or_var_op_name2pull_job_name[
            op_name
        ]
        self.LaunchJob(
            job_instance_util.MakePullJobInstance(pull_job_name, op_name, pull_data_cb)
        )

    def HasAnyCallbackAfterFunctionReturn(self):
        return len(self.uuid2watch_handler) > 0

    def StashVariableBlob4Job(self, job_name, var_name, var_blob):
        if var_name not in self.var_name2var_blob_:
            self.var_name2var_blob_[var_name] = var_blob
        if job_name not in self.job_name2var_name2var_blob_:
            self.job_name2var_name2var_blob_[job_name] = dict()
        assert var_name not in self.job_name2var_name2var_blob_[job_name]
        self.job_name2var_name2var_blob_[job_name][var_name] = var_blob

    # return global_variable_blob, job_variable_blob
    def TryGetVariableBlobOfJobFromStash(self, job_name, var_name):
        if var_name not in self.var_name2var_blob_:
            return None, None
        global_variable_blob = self.var_name2var_blob_[var_name]
        if job_name not in self.job_name2var_name2var_blob_:
            return global_variable_blob, None

        var_name2var_blob = self.job_name2var_name2var_blob_[job_name]
        if var_name not in var_name2var_blob:
            return global_variable_blob, None
        assert global_variable_blob is var_name2var_blob[var_name]
        return global_variable_blob, var_name2var_blob[var_name]

    def CurrentEagerGlobalFunctionDesc(self):
        if len(self.eager_global_function_desc_stack_) == 0:
            return None
        return self.eager_global_function_desc_stack_[0]

    @contextmanager
    def _EagerGlobalFunctionDescScope(self, function_desc):
        assert len(self.backward_blob_register.blob_name2object) == 0
        assert len(self.job_name2var_name2var_blob_) == 0
        self.eager_global_function_desc_stack_.insert(0, function_desc)
        try:
            yield
        finally:
            self.job_name2var_name2var_blob_ = dict()
            self.eager_global_function_desc_stack_.pop(0)
            keys = list(self.backward_blob_register.blob_name2object.keys())
            for key in keys:
                del self.backward_blob_register.blob_name2object[key]

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


@enable_if.condition(hob.in_normal_mode & ~hob.any_global_function_defined)
def enable_eager_execution(val=True):
    return c_api_util.EnableEagerExecution(val)


@oneflow_export("enable_eager_execution")
def api_enable_eager_execution(val=True):
    return enable_if.unique([enable_eager_execution])(val)


@oneflow_export("eager_execution_enabled")
def api_eager_execution_enabled():
    return c_api_util.EagerExecutionEnabled()


@oneflow_export("clear_default_session")
def clear_default_session():
    r"""Clear the default session. All compiled OneFlow functions will be deleted.

    """
    session_ctx.TryCloseDefaultSession()
    session_ctx.OpenDefaultSession(Session())
    c_api_util.EnableEagerExecution(False)


@oneflow_export("scope.current_scope")
def current_scope():
    r""" Return current scope
    """
    job_name = oneflow.current_global_function_desc().job_config_proto.job_name
    return session_ctx.GetDefaultSession().GetCurrentScope(job_name)


@oneflow_export("sync_default_session")
def sync_default_session():
    r"""Synchronize the default session. Block until every synchronous OneFlow function and its callback finishes running.

    """
    session_ctx.GetDefaultSession().Sync()


def _TryCompleteConfigProto(config_proto):
    if config_proto.resource.machine_num == 0:
        config_proto.resource.machine_num = len(env_util.default_env_proto.machine)


def _GetDefaultConfigProto():
    config_proto = job_set_util.ConfigProto()
    config_proto.resource.machine_num = 0
    config_proto.resource.gpu_device_num = 1
    config_proto.io_conf.data_fs_conf.localfs_conf.SetInParent()
    config_proto.io_conf.snapshot_fs_conf.localfs_conf.SetInParent()
    return config_proto


session_ctx.OpenDefaultSession(Session())
