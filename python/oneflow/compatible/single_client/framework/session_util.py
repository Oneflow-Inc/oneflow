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
import inspect
import threading
import traceback
from contextlib import contextmanager
from typing import Callable

from google.protobuf import text_format

import oneflow._oneflow_internal
from oneflow import oneflow_deprecate
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.eager import op_executor as op_executor
from oneflow.compatible.single_client.experimental import interface_op_read_and_write
from oneflow.compatible.single_client.framework import c_api_util as c_api_util
from oneflow.compatible.single_client.framework import check_point_v2 as check_point_v2
from oneflow.compatible.single_client.framework import compiler as compiler
from oneflow.compatible.single_client.framework import config_util as config_util
from oneflow.compatible.single_client.framework import env_util as env_util
from oneflow.compatible.single_client.framework import hob as hob
from oneflow.compatible.single_client.framework import job_instance as job_instance_util
from oneflow.compatible.single_client.framework import module as module_util
from oneflow.compatible.single_client.framework import push_util as push_util
from oneflow.compatible.single_client.framework import session_context as session_ctx
from oneflow.compatible.single_client.framework import typing_util as oft_util
from oneflow.compatible.single_client.framework.check_point import SnapshotManager
from oneflow.compatible.single_client.framework.function_desc import FunctionDesc
from oneflow.compatible.single_client.framework.pull_util import (
    EagerFutureRemoteBlobs,
    LazyFutureRemoteBlobs,
)
from oneflow.compatible.single_client.framework.session_context import SessionStatus
from oneflow.compatible.single_client.support import enable_if as enable_if
from oneflow.core.job import job_set_pb2 as job_set_util
from oneflow.core.job.job_set_pb2 import ConfigProto


class Session(object):
    def __init__(self, sess_id):
        self.job_name2function_desc_ = {}
        self.job_name2job_ = {}
        self.status_ = SessionStatus.OPEN
        self.cond_var_ = threading.Condition()
        self.running_job_cnt_ = 0
        self.inter_user_job_info_ = None
        self.uuid2watch_handler_ = {}
        self.config_proto_ = None
        self.resource_ = None
        self.job_name2var_name2var_blob_ = {}
        self.job_name2module_name2module_ = {}
        self.existed_module_names_ = set()
        self.var_name2var_blob_ = {}
        self.interface_op_name2op_attr_ = {}
        self.interface_op_name2job_name_ = {}
        self.lazy_interface_op_name2parallel_conf_ = {}
        self.op_name2lazy_blob_cache_ = {}
        self.job_name2name_scope_stack_ = {}
        self.eager_global_function_desc_stack_ = []
        self.function_flag_name2default_val_ = {}
        self._UpdateFunctionFlagName2DefaultVal()
        self.scope_attr_name2default_val_ = {}
        self._UpdateScopeAttrName2DefaultVal()
        self.sess_ = oneflow._oneflow_internal.RegsiterSession(sess_id)
        self.backward_blob_register_ = oneflow._oneflow_internal.BlobRegister()
        self.snapshot_mgr_ = SnapshotManager()
        self.eager_config_proto_ctx_ = None

    @property
    def id(self):
        return self.sess_.id

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
    def resource(self):
        if self.resource_ is None:
            return flow.env.current_resource()
        else:
            return self.resource_

    @property
    def uuid2watch_handler(self):
        return self.uuid2watch_handler_

    @property
    def function_flag_name2default_val(self):
        return self.function_flag_name2default_val_

    @property
    def scope_attr_name2default_val(self):
        return self.scope_attr_name2default_val_

    @property
    def inter_user_job_info(self):
        return self.inter_user_job_info_

    @property
    def job_name2name_scope_stack(self):
        return self.job_name2name_scope_stack_

    @property
    def backward_blob_register(self):
        return self.backward_blob_register_

    @property
    def snapshot_mgr(self):
        return self.snapshot_mgr_

    @property
    def var_name2var_blob(self):
        return self.var_name2var_blob_

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
        items = c_api_util.GetFunctionConfigDef().attr_name2attr_def.items()
        self.function_flag_name2default_val_ = {k: v.default_val for (k, v) in items}

    def _UpdateScopeAttrName2DefaultVal(self):
        items = c_api_util.GetScopeConfigDef().attr_name2attr_def.items()
        self.scope_attr_name2default_val_ = {k: v.default_val for (k, v) in items}

    def TryInit(self):
        if self.status_ is SessionStatus.OPEN:
            self.Init()
        return self

    def UpdateInfo4InterfaceOp(self):
        for op_attr in c_api_util.GetInterfaceOpAttributes().op_attribute:
            self.interface_op_name2op_attr_[op_attr.op_conf.name] = op_attr
        for job in c_api_util.GetJobSet().job:
            op_name2parallel_conf = {}
            for placement_group in job.placement.placement_group:
                for op_name in placement_group.op_set.op_name:
                    op_name2parallel_conf[op_name] = placement_group.parallel_conf
            for op_conf in job.net.op:
                if c_api_util.IsInterfaceOpConf(op_conf):
                    self.interface_op_name2job_name_[
                        op_conf.name
                    ] = job.job_conf.job_name
                    self.lazy_interface_op_name2parallel_conf_[
                        op_conf.name
                    ] = op_name2parallel_conf[op_conf.name]

    def Init(self):
        assert self.status_ is SessionStatus.OPEN
        self.status_ = SessionStatus.RUNNING
        if not oneflow._oneflow_internal.IsEnvInited():
            flow.env.init()
        _TryCompleteConfigProto(self.config_proto)
        self.resource_ = self.config_proto.resource
        if not oneflow._oneflow_internal.EagerExecutionEnabled():
            c_api_util.InitLazyGlobalSession(self.config_proto)
            for (job_name, func_desc) in self.job_name2function_desc_.items():
                compiler.Compile(self, func_desc, self.config_proto)
                self.existed_module_names_ = set()
            self.job_name2var_name2var_blob_ = dict()
            assert len(self.job_name2function_desc_.items()) > 0
            oneflow._oneflow_internal.StartLazyGlobalSession()
            self.inter_user_job_info_ = c_api_util.GetInterUserJobInfo()
            self.UpdateInfo4InterfaceOp()
            if not config_util.api_legacy_model_io_enabled():
                check_point_v2.Init()
        else:
            self.eager_config_proto_ctx_ = oneflow._oneflow_internal.LogicalConfigProtoContext(
                str(self.config_proto)
            )
        return self

    def FindOrCreateLazyBlob(self, op_name, Create):
        if op_name not in self.op_name2lazy_blob_cache_:
            self.op_name2lazy_blob_cache_[op_name] = Create()
        return self.op_name2lazy_blob_cache_[op_name]

    def TryClose(self):
        if self.status_ is SessionStatus.RUNNING:
            self.Close()
        if self.status_ != SessionStatus.CLOSED:
            oneflow._oneflow_internal.ClearSessionById(self.id)
        self.status_ = SessionStatus.CLOSED

    def Close(self):
        assert self.status_ is SessionStatus.RUNNING
        self.Sync()
        assert len(self.job_name2var_name2var_blob_) == 0
        del self.var_name2var_blob_
        del self.job_name2module_name2module_
        self.ReleaseLazyRefBlob()
        self.ForceReleaseEagerBlobs()
        oneflow._oneflow_internal.StopLazyGlobalSession()
        oneflow._oneflow_internal.DestroyLazyGlobalSession()
        self.resource_ = None
        if self.eager_config_proto_ctx_:
            del self.eager_config_proto_ctx_

    def AddJob(self, function_desc):
        assert self.status_ is SessionStatus.OPEN
        assert isinstance(function_desc, FunctionDesc)
        self.job_name2function_desc_[function_desc.job_func.__name__] = function_desc

    def StashJob(self, job_name=None, key=None):
        assert self.status_ is SessionStatus.RUNNING, "current status {}".format(
            self.status_
        )
        job = c_api_util.GetCurrentJob()
        if job_name is not None:
            assert (
                job.job_conf.job_name == job_name
            ), "{} is not current job name".format(job_name)
        else:
            job_name = job.job_conf.job_name
        if key is None:
            key = job_name
        self.job_name2job_[key] = job

    def Job(self, job_name):
        assert self.status_ is SessionStatus.RUNNING
        if job_name not in self.job_name2job_:
            return None
        return self.job_name2job_[job_name]

    def Sync(self):
        assert self.status_ is SessionStatus.RUNNING
        self.cond_var_.acquire()
        while self.running_job_cnt_ > 0:
            self.cond_var_.wait()
        assert self.running_job_cnt_ == 0
        self.cond_var_.release()

    def ReleaseLazyRefBlob(self):
        self.op_name2lazy_blob_cache_.clear()

    def ForceReleaseEagerBlobs(self):
        oneflow._oneflow_internal.GetDefaultBlobRegister().ForceReleaseAll()
        self.backward_blob_register_.ForceReleaseAll()

    def LazyRun(self, job_func, *arg):
        assert self.status_ is SessionStatus.RUNNING
        remote_blobs = self.LaunchUserJob(job_func, *arg)
        if remote_blobs is None:
            return
        future_blob = LazyFutureRemoteBlobs(self).SetResult(remote_blobs).Inited()
        annotation = inspect.signature(job_func).return_annotation
        return oft_util.TransformGlobalFunctionResult(future_blob, annotation)

    def EagerRun(self, function_desc, *arg):
        with self._EagerGlobalFunctionDescScope(function_desc):
            remote_blobs = compiler.EagerRun(
                self, function_desc, self.config_proto, arg
            )
            if remote_blobs is None:
                return
            future_blob = EagerFutureRemoteBlobs().SetResult(remote_blobs).Inited()
        annotation = inspect.signature(function_desc.job_func).return_annotation
        return oft_util.TransformGlobalFunctionResult(future_blob, annotation)

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
        oneflow._oneflow_internal.LaunchJob(job_instance)

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

    def AddInfo4InterfaceOpName(self, interface_op_name, op_attribute):
        if flow.eager_execution_enabled():
            self.interface_op_name2op_attr_[interface_op_name] = op_attribute
            self.interface_op_name2job_name_[
                interface_op_name
            ] = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
        else:
            pass

    def OpAttribute4InterfaceOpName(self, interface_op_name):
        return self.interface_op_name2op_attr_[interface_op_name]

    def ParallelConf4LazyInterfaceOpName(self, interface_op_name):
        return self.lazy_interface_op_name2parallel_conf_[interface_op_name]

    def JobName4InterfaceOpName(self, interface_op_name):
        return self.interface_op_name2job_name_[interface_op_name]

    @property
    def interface_ops(self):
        return self.interface_op_name2op_attr_.keys()

    def TryGetVariableBlobOfJobFromStash(self, job_name, var_name):
        if var_name not in self.var_name2var_blob_:
            return (None, None)
        global_variable_blob = self.var_name2var_blob_[var_name]
        if job_name not in self.job_name2var_name2var_blob_:
            return (global_variable_blob, None)
        var_name2var_blob = self.job_name2var_name2var_blob_[job_name]
        if var_name not in var_name2var_blob:
            return (global_variable_blob, None)
        return (global_variable_blob, var_name2var_blob[var_name])

    def CurrentEagerGlobalFunctionDesc(self):
        if len(self.eager_global_function_desc_stack_) == 0:
            return None
        return self.eager_global_function_desc_stack_[0]

    def has_empty_is_mirrored_strategy_enabled_stack(self):
        return self.sess_.is_mirrored_strategy_enabled_stack_size() == 0

    def push_mirrored_strategy_enabled(self, val):
        assert isinstance(val, bool)
        self.sess_.push_mirrored_strategy_enabled(val)

    def pop_mirrored_strategy_enabled(self):
        self.sess_.pop_mirrored_strategy_enabled()

    def is_mirrored_strategy_enabled(self):
        return self.sess_.is_mirrored_strategy_enabled()

    def is_consistent_strategy_enabled(self):
        return self.sess_.is_consistent_strategy_enabled()

    @contextmanager
    def _EagerGlobalFunctionDescScope(self, function_desc):
        assert len(self.backward_blob_register.blob_name2object) == 0
        assert len(self.job_name2var_name2var_blob_) == 0
        self.eager_global_function_desc_stack_.insert(0, function_desc)
        try:
            yield
        finally:
            self.existed_module_names_ = set()
            self.job_name2var_name2var_blob_ = dict()
            self.eager_global_function_desc_stack_.pop(0)
            keys = list(dict(self.backward_blob_register.blob_name2object).keys())
            for key in keys:
                self.backward_blob_register.ClearObject4BlobName(key)

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

    def __del__(self):
        self.TryClose()


def api_find_or_create_module(
    module_name: str, create: Callable[[], None], reuse: bool = False
):
    func = enable_if.unique([find_or_create_module])
    return func(module_name, create, reuse)


@enable_if.condition(hob.in_global_mode)
def find_or_create_module(module_name, create, reuse=False):
    assert callable(create)
    sess = session_ctx.GetDefaultSession()
    job_name = flow.current_global_function_desc().job_config_proto.job_name()
    if job_name not in sess.job_name2module_name2module_:
        sess.job_name2module_name2module_[job_name] = {}
    module_name2module = sess.job_name2module_name2module_[job_name]
    if module_name not in module_name2module:
        module = create()
        assert isinstance(module, module_util.Module)
        module_name2module[module_name] = module
    elif not reuse:
        assert module_name not in sess.existed_module_names_, (
            "duplicated module_name `%s' in global_function `%s'"
            % (module_name, job_name)
        )
    else:
        pass
    sess.existed_module_names_.add(module_name)
    return module_name2module[module_name]


def api_eager_execution_enabled() -> bool:
    """Get current setting of the job, if enable eager execution mode ,then return True

    Returns:
        bool: [description]
    """
    return oneflow._oneflow_internal.EagerExecutionEnabled()


def api_clear_default_session() -> None:
    """Clear the default session. All compiled OneFlow functions will be deleted.
    """
    func = enable_if.unique([clear_default_session])
    return func()


@enable_if.condition(hob.in_normal_mode)
def clear_default_session():
    session_ctx.TryCloseDefaultSession()
    session_ctx.OpenDefaultSession(Session(oneflow._oneflow_internal.NewSessionId()))


def api_sync_default_session() -> None:
    """Synchronize the default session. Block until every synchronous OneFlow function and its callback finishes running.
    """
    func = enable_if.unique([sync_default_session])
    return func()


@enable_if.condition(hob.in_normal_mode)
def sync_default_session() -> None:
    session_ctx.GetDefaultSession().Sync()


def _TryCompleteConfigProto(config_proto):
    if config_proto.resource.machine_num == 0:
        config_proto.resource.machine_num = oneflow._oneflow_internal.GetNodeSize()


def _GetDefaultConfigProto():
    config_proto = job_set_util.ConfigProto()
    config_proto.resource.machine_num = 0
    if oneflow._oneflow_internal.flags.with_cuda():
        config_proto.resource.gpu_device_num = 1
    else:
        config_proto.resource.cpu_device_num = 1
        config_proto.resource.gpu_device_num = 0
    config_proto.session_id = session_ctx.GetDefaultSession().id
    return config_proto


def TmpInitEagerGlobalSession():
    config_pb = _GetDefaultConfigProto()
    config_proto_str = text_format.MessageToString(config_pb)
    oneflow._oneflow_internal.InitEagerGlobalSession(config_proto_str)
