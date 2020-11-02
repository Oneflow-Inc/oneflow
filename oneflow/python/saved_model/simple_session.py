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
import oneflow as flow
import oneflow.core.job.job_conf_pb2 as job_conf_pb
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_ctx
import oneflow.python.framework.session_util as session_util
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.scope_util as scope_util
import oneflow.python.framework.runtime_mode as runtime_mode
import oneflow.python.framework.job_instance as job_instance_util
import oneflow.python.framework.input_blob_def as input_blob_util
import oneflow.python.framework.dtype as dtype_util
import contextlib
import threading
import inspect
import numpy as np

from oneflow.python.oneflow_export import oneflow_export


class OutputFuture(object):
    def __init__(self):
        self.cond_var_ = threading.Condition()
        self.done_ = False
        self.result_ = None

    def set_result(self, result):
        self.cond_var_.acquire()
        self.result_ = result
        self.done_ = True
        self.cond_var_.notify()
        self.cond_var_.release()

    def get(self):
        self.cond_var_.acquire()
        while self.done_ is False:
            self.cond_var_.wait()
        self.cond_var_.release()
        return self.result_


@oneflow_export("SimpleSession")
class SimpleSession(object):
    # TODO: support multi user job
    # TODO: check multi user job input/ouput name conflict

    class SessionStatus:
        OPEN = "OPEN"
        RUNNING = "RUNNING"
        CLOSED = "CLOSED"

    def __init__(self, config_proto=None):
        self.cond_var_ = threading.Condition()
        self.running_job_cnt_ = 0
        self.is_mirrored_ = False
        self.checkpoint_path_ = None
        self.job_name2job_conf_ = {}
        # self.job_name2inter_user_job_info_ = {}
        self.inter_user_job_info_ = None
        self.job_name2input_name2lbn_ = {}
        self.job_name2output_name2lbn_ = {}
        self.output_name2future_ = {}

        # env init
        if not c_api_util.IsEnvInited():
            flow.env.init()

        # session init
        if config_proto is None:
            self.config_proto_ = session_util._GetDefaultConfigProto()
        else:
            self.config_proto_ = config_proto

        session_util._TryCompleteConfigProto(self.config_proto_)
        c_api_util.InitLazyGlobalSession(self.config_proto_)
        self.status_ = self.SessionStatus.OPEN

    def __del__(self):
        self.close()

    def close(self):
        self._sync()
        if self.status_ == self.SessionStatus.RUNNING:
            c_api_util.StopLazyGlobalSession()
            c_api_util.DestroyLazyGlobalSession()
            c_api_util.DestroyEnv()
        elif self.status_ == self.SessionStatus.OPEN:
            c_api_util.DestroyLazyGlobalSession()
            c_api_util.DestroyEnv()
        else:
            pass

        self.status_ = self.SessionStatus.CLOSED

    def _check_status(self, *status):
        check_sucess = False
        for stat in status:
            if self.status_ == stat:
                check_sucess = True
                break

        if check_sucess is False:
            caller_func_name = inspect.stack()[1].function
            allowed_status = ",".join(status)
            raise ValueError(
                "The calling to {} is only allowed when status is {}, current status is {}".format(
                    caller_func_name, allowed_status, self.status_
                )
            )

    def set_checkpoint_path(self, checkpoint_path):
        self._check_status(self.SessionStatus.OPEN)
        self.checkpoint_path_ = checkpoint_path

    def setup_job_signature(self, job_name, signature):
        self._check_status(self.SessionStatus.OPEN)
        if job_name in self.job_name2job_conf_:
            job_conf = self.job_name2job_conf_[job_name]
            assert job_conf.job_name == job_name
        else:
            job_conf = job_conf_pb.JobConfigProto()
            job_conf.job_name = job_name
            self.job_name2job_conf_[job_name] = job_conf

        job_conf.signature.CopyFrom(signature)

        self.job_name2input_name2lbn_[job_name] = {}
        for input_name, input_def in signature.inputs.items():
            self.job_name2input_name2lbn_[job_name][input_name] = "{}/{}".format(
                input_def.lbi.op_name, input_def.lbi.blob_name
            )

        self.job_name2output_name2lbn_[job_name] = {}
        for output_name, output_def in signature.outputs.items():
            self.job_name2output_name2lbn_[job_name][output_name] = "{}/{}".format(
                output_def.lbi.op_name, output_def.lbi.blob_name
            )

    def _get_job_conf(self, job_name):
        if job_name in self.job_name2job_conf_:
            return self.job_name2job_conf_[job_name]
        else:
            job_conf = job_conf_pb.JobConfigProto()
            job_conf.job_name = job_name
            self.job_name2job_conf_[job_name] = job_conf
            return job_conf

    @contextlib.contextmanager
    def open(self, job_name):
        self._check_status(self.SessionStatus.OPEN)
        c_api_util.JobBuildAndInferCtx_Open(job_name)
        job_conf = self._get_job_conf(job_name)
        c_api_util.CurJobBuildAndInferCtx_SetJobConf(job_conf)

        tag_and_dev_ids = placement_util.GetDefaultMachineDeviceIds(
            self.config_proto_.resource
        )
        scope = scope_util.MakeInitialScope(
            job_conf, *tag_and_dev_ids, self.is_mirrored_
        )

        with runtime_mode.ModeScope(runtime_mode.GLOBAL_MODE):
            with scope_util.ScopeContext(scope):
                yield self

        c_api_util.JobBuildAndInferCtx_Close()

    def compile(self, op_list):
        self._check_status(self.SessionStatus.OPEN)
        for op_conf in op_list:
            compile_ctx.CurJobAddOp(op_conf)

        c_api_util.CurJobBuildAndInferCtx_Complete()

    def launch(self):
        self._check_status(self.SessionStatus.OPEN)
        c_api_util.StartLazyGlobalSession()
        self.inter_user_job_info_ = c_api_util.GetInterUserJobInfo()
        self._run_load_checkpoint_job()
        self.status_ = self.SessionStatus.RUNNING

    # TODO: list jobs

    def list_inputs(self):
        self._check_status(self.SessionStatus.RUNNING)
        input_names = []
        for (
            input_name,
            _,
        ) in self.inter_user_job_info_.input_or_var_op_name2push_job_name.items():
            input_names.append(input_name)
        return tuple(input_names)

    def list_outputs(self):
        self._check_status(self.SessionStatus.RUNNING)
        output_names = []
        for (
            output_name,
            _,
        ) in self.inter_user_job_info_.output_or_var_op_name2pull_job_name.items():
            output_names.append(output_name)
        return tuple(output_names)

    def _query_input_lbn(self, input_name):
        input_job_name = None
        input_lbn = None
        for job_name, input_name2lbn in self.job_name2input_name2lbn_.items():
            if input_name in input_name2lbn:
                input_lbn = input_name2lbn[input_name]
                input_job_name = job_name

        return input_lbn, input_job_name

    def _query_output_lbn(self, output_name):
        output_job_name = None
        output_lbn = None
        for job_name, output_name2lbn in self.job_name2output_name2lbn_.items():
            if output_name in output_name2lbn:
                output_lbn = output_name2lbn[output_name]
                output_job_name = job_name

        return output_lbn, output_job_name

    def input_info(self, input_name):
        self._check_status(self.SessionStatus.OPEN, self.SessionStatus.RUNNING)
        input_lbn, input_job_name = self._query_input_lbn(input_name)
        if input_job_name is None or input_lbn is None:
            raise ValueError('can not find input "{}"'.format(input_name))

        input_shape = c_api_util.JobBuildAndInferCtx_GetStaticShape(
            input_job_name, input_lbn
        )
        input_dtype = c_api_util.JobBuildAndInferCtx_GetDataType(
            input_job_name, input_lbn
        )
        input_dtype = dtype_util.convert_proto_dtype_to_oneflow_dtype(input_dtype)
        input_dtype = dtype_util.convert_oneflow_dtype_to_numpy_dtype(input_dtype)
        # TODO: other info
        return {"shape": input_shape, "dtype": input_dtype}

    def run(self, job_name, **kwargs):
        # TODO: check args and warn unexpected args
        self._check_status(self.SessionStatus.RUNNING)
        self._run_push_jobs(**kwargs)
        for job_name in self.job_name2job_conf_.keys():
            job_inst = job_instance_util.MakeUserJobInstance(job_name)
            self._run_job(job_inst)
        self._run_pull_jobs()

        # process result
        return tuple(
            self.output_name2future_[output_name].get()
            for output_name in self.list_outputs()
        )

    def _run_job(self, job_inst):
        self._increase_running_job_cnt()
        job_inst.AddPostFinishCallback(lambda _: self._decrease_running_job_cnt())
        c_api_util.LaunchJob(job_inst)

    def _run_push_jobs(self, **kwargs):
        for (
            input_name,
            push_job_name,
        ) in self.inter_user_job_info_.input_or_var_op_name2push_job_name.items():
            if input_name not in kwargs:
                raise ValueError('input "{}" is absent'.format(input_name))

            input_numpy = kwargs[input_name]
            if not isinstance(input_numpy, np.ndarray):
                raise ValueError('input "{}" requires numpy.ndarray'.format(input_name))

            push_fn = input_blob_util._MakePushNdarrayCallback(input_numpy)
            push_job_inst = job_instance_util.MakePushJobInstance(
                push_job_name, input_name, push_fn
            )
            self._run_job(push_job_inst)

    def _run_pull_jobs(self):
        for (
            output_name,
            pull_job_name,
        ) in self.inter_user_job_info_.output_or_var_op_name2pull_job_name.items():
            self.output_name2future_[output_name] = OutputFuture()
            pull_fn = self._make_pull_job_cb(output_name)
            pull_job_inst = job_instance_util.MakePullJobInstance(
                pull_job_name, output_name, pull_fn
            )
            self._run_job(pull_job_inst)

    def _make_pull_job_cb(self, output_name):
        future = self.output_name2future_[output_name]
        output_lbn, output_job_name = self._query_output_lbn(output_name)
        split_axis = c_api_util.JobBuildAndInferCtx_GetSplitAxisFromProducerView(
            output_job_name, output_lbn
        )

        def pull_fn(ofblob):
            ndarray_lists = ofblob.CopyToNdarrayLists()
            assert len(ndarray_lists) == 1
            ndarray_list = ndarray_lists[0]
            if len(ndarray_list) == 1:
                future.set_result(ndarray_list[0])
            else:
                assert split_axis is not None
                future.set_result(np.concatenate(ndarray_list, axis=split_axis))

        return pull_fn

    def _run_load_checkpoint_job(self):
        if self.checkpoint_path_ is None:
            raise ValueError("checkpoint path not set")

        def copy_model_load_path(ofblob):
            ofblob.CopyFromNdarray(
                np.frombuffer(self.checkpoint_path_.encode("ascii"), dtype=np.int8)
            )

        load_checkpoint_job_inst = job_instance_util.MakeJobInstance(
            self.inter_user_job_info_.global_model_load_job_name,
            push_cb=copy_model_load_path,
            finish_cb=lambda: None,
        )
        self._run_job(load_checkpoint_job_inst)

    def _sync(self):
        self.cond_var_.acquire()
        while self.running_job_cnt_ > 0:
            self.cond_var_.wait()
        assert self.running_job_cnt_ == 0
        self.cond_var_.release()

    def _increase_running_job_cnt(self):
        self.cond_var_.acquire()
        self.running_job_cnt_ += 1
        self.cond_var_.release()

    def _decrease_running_job_cnt(self):
        self.cond_var_.acquire()
        self.running_job_cnt_ -= 1
        self.cond_var_.notify()
        self.cond_var_.release()

    def print_job_set(self):
        self._check_status(self.SessionStatus.OPEN, self.SessionStatus.RUNNING)
        job_set = c_api_util.GetJobSet()
        for job in job_set.job:
            print("job_name:", job.job_conf.job_name)
            for op_conf in job.net.op:
                print("\top_name:", op_conf.name)
