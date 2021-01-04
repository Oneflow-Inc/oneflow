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
import contextlib
import threading
import inspect
import asyncio
import numpy as np

import oneflow as flow
import oneflow_api
import oneflow_api.oneflow.core.job.job_conf as job_conf_proto_cfg
import oneflow_api.oneflow.core.framework.tensor as tensor_proto_cfg
import oneflow_api.oneflow.core.common.shape as shape_proto_cfg
import oneflow_api.oneflow.core.common.data_type as dtype_proto_cfg
import oneflow.core.job.job_conf_pb2 as job_conf_proto
import oneflow.core.framework.tensor_pb2 as tensor_proto
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_ctx
import oneflow.python.framework.session_util as session_util
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.scope_util as scope_util
import oneflow.python.framework.runtime_mode as runtime_mode
import oneflow.python.framework.job_instance as job_instance_util
import oneflow.python.framework.input_blob_def as input_blob_util
import oneflow.python.framework.dtype as dtype_util
from oneflow.python.oneflow_export import oneflow_export


def _need_check_device_tag(op_conf):
    if op_conf.HasField("return_conf"):
        return False

    return op_conf.HasField("device_tag")


def _signature_proto_to_cfg(signature_proto, mut_signature_cfg):
    assert isinstance(signature_proto, job_conf_proto.JobSignatureDef)
    assert isinstance(mut_signature_cfg, job_conf_proto_cfg.JobSignatureDef)

    for input_name, input_def in signature_proto.inputs.items():
        input_def_cfg = job_conf_proto_cfg.JobInputDef()
        input_def_cfg.mutable_lbi().set_op_name(input_def.lbi.op_name)
        input_def_cfg.mutable_lbi().set_blob_name(input_def.lbi.blob_name)
        _inferface_blob_conf_proto_to_cfg(
            input_def.blob_conf, input_def_cfg.mutable_blob_conf()
        )
        mut_signature_cfg.mutable_inputs()[input_name].CopyFrom(input_def_cfg)

    for output_name, output_def in signature_proto.outputs.items():
        output_def_cfg = job_conf_proto_cfg.JobOutputDef()
        output_def_cfg.mutable_lbi().set_op_name(output_def.lbi.op_name)
        output_def_cfg.mutable_lbi().set_blob_name(output_def.lbi.blob_name)
        mut_signature_cfg.mutable_outputs()[output_name].CopyFrom(output_def_cfg)


def _inferface_blob_conf_proto_to_cfg(
    inferface_blob_conf_proto, mut_inferface_blob_conf_cfg
):
    assert isinstance(inferface_blob_conf_proto, tensor_proto.InterfaceBlobConf)
    assert isinstance(mut_inferface_blob_conf_cfg, tensor_proto_cfg.InterfaceBlobConf)

    shape = shape_proto_cfg.ShapeProto()
    for dim in inferface_blob_conf_proto.shape.dim:
        shape.add_dim(dim)

    mut_inferface_blob_conf_cfg.mutable_shape().CopyFrom(shape)
    dtype = dtype_proto_cfg.DataType(int(inferface_blob_conf_proto.data_type))
    mut_inferface_blob_conf_cfg.set_data_type(dtype)

    split_axis = dtype_proto_cfg.OptInt64()
    if inferface_blob_conf_proto.split_axis.HasField("value"):
        split_axis.set_value(inferface_blob_conf_proto.split_axis.value)
    mut_inferface_blob_conf_cfg.mutable_split_axis().CopyFrom(split_axis)

    batch_axis = dtype_proto_cfg.OptInt64()
    if inferface_blob_conf_proto.batch_axis.HasField("value"):
        batch_axis.set_value(inferface_blob_conf_proto.batch_axis.value)
    mut_inferface_blob_conf_cfg.mutable_batch_axis().CopyFrom(batch_axis)

    mut_inferface_blob_conf_cfg.set_is_dynamic(inferface_blob_conf_proto.is_dynamic)
    mut_inferface_blob_conf_cfg.set_is_tensor_list(
        inferface_blob_conf_proto.is_tensor_list
    )


@oneflow_export("serving.SessionOption")
class SessionOption(object):
    def __init__(self):
        self.device_tag = "gpu"
        self.device_num = 1


@oneflow_export("serving.InferenceSession")
class InferenceSession(object):
    class SessionStatus:
        OPEN = "OPEN"
        RUNNING = "RUNNING"
        CLOSED = "CLOSED"

    def __init__(self, option=None):
        if option is None:
            self.option_ = SessionOption()
        else:
            assert isinstance(option, SessionOption)
            self.option_ = option

        self.cond_var_ = threading.Condition()
        self.running_job_cnt_ = 0
        self.is_mirrored_ = False
        self.checkpoint_path_ = None
        self.config_proto_ = None
        self.job_name2job_conf_ = {}
        self.inter_user_job_info_ = None
        self.job_name2input_name2lbn_ = {}
        self.job_name2output_name2lbn_ = {}
        self.output_name2future_ = {}
        self.status_ = None

        self.init()

    def __del__(self):
        self.close()

    def init(self):
        # env init
        if not oneflow_api.IsEnvInited():
            flow.env.init()

        # session init
        if not oneflow_api.IsSessionInited():
            self._make_config_proto()
            session_util._TryCompleteConfigProto(self.config_proto_)
            c_api_util.InitLazyGlobalSession(self.config_proto_)

        self.status_ = self.SessionStatus.OPEN

    def _make_config_proto(self):
        if self.config_proto_ is None:
            self.config_proto_ = session_util._GetDefaultConfigProto()

        if self.option_.device_tag == "gpu":
            self.config_proto_.resource.gpu_device_num = self.option_.device_num
        elif self.option_.device_tag == "cpu":
            self.config_proto_.resource.cpu_device_num = self.option_.device_num
            self.config_proto_.resource.gpu_device_num = 0
        else:
            raise NotImplementedError(
                "not supported device tag {}".format(self.option_.device_tag)
            )

        self.config_proto_.io_conf.enable_legacy_model_io = True

    def close(self):
        self._sync()
        if self.status_ == self.SessionStatus.RUNNING:
            oneflow_api.StopLazyGlobalSession()
            oneflow_api.DestroyLazyGlobalSession()
            oneflow_api.DestroyEnv()
        elif self.status_ == self.SessionStatus.OPEN:
            oneflow_api.DestroyLazyGlobalSession()
            oneflow_api.DestroyEnv()
        else:
            pass

        self.status_ = self.SessionStatus.CLOSED

    def _check_status(self, *status):
        check_success = False
        for stat in status:
            if self.status_ == stat:
                check_success = True
                break

        if check_success is False:
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

    def set_job_signature(self, job_name, signature):
        assert isinstance(signature, job_conf_proto.JobSignatureDef)
        job_conf = self._get_job_conf(job_name)
        _signature_proto_to_cfg(signature, job_conf.mutable_signature())

        self.job_name2input_name2lbn_[job_name] = {}
        for input_name, input_def in signature.inputs.items():
            lbn = "{}/{}".format(input_def.lbi.op_name, input_def.lbi.blob_name)
            self.job_name2input_name2lbn_[job_name][input_name] = lbn

        self.job_name2output_name2lbn_[job_name] = {}
        for output_name, output_def in signature.outputs.items():
            lbn = "{}/{}".format(output_def.lbi.op_name, output_def.lbi.blob_name)
            self.job_name2output_name2lbn_[job_name][output_name] = lbn

    def _get_job_conf(self, job_name):
        if job_name in self.job_name2job_conf_:
            return self.job_name2job_conf_[job_name]
        else:
            job_conf = job_conf_proto_cfg.JobConfigProto()
            job_conf.set_job_name(job_name)
            self.job_name2job_conf_[job_name] = job_conf
            return job_conf

    @contextlib.contextmanager
    def open(self, job_name, signature):
        self._check_status(self.SessionStatus.OPEN)
        c_api_util.JobBuildAndInferCtx_Open(job_name)

        self.set_job_signature(job_name, signature)
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

        oneflow_api.JobBuildAndInferCtx_Close()

    def compile(self, op_list):
        self._check_status(self.SessionStatus.OPEN)
        scope = flow.current_scope()
        device_tag = scope.device_parallel_desc_symbol.device_tag
        for op_conf in op_list:
            if _need_check_device_tag(op_conf) and op_conf.device_tag != device_tag:
                print(
                    "WARNING: the device_tag of op {} is not equal to the device_tag of seesion's current scope"
                    " ({} vs. {})"
                    ", which may cause the op graph to be incompatible".format(
                        op_conf.name, op_conf.device_tag, device_tag
                    )
                )

            compile_ctx.CurJobAddOp(op_conf)

        oneflow_api.CurJobBuildAndInferCtx_Complete()

    def launch(self):
        self._check_status(self.SessionStatus.OPEN)
        oneflow_api.StartLazyGlobalSession()
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
        self._check_status(self.SessionStatus.RUNNING)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_run(job_name, **kwargs))

    async def async_run(self, job_name, **kwargs):
        self._check_status(self.SessionStatus.RUNNING)
        self._run_push_jobs(**kwargs)
        job_inst = job_instance_util.MakeUserJobInstance(job_name)
        self._run_job(job_inst)
        output_futures = tuple(self._run_pull_jobs().values())
        return await asyncio.gather(*output_futures)

    def _run_job(self, job_inst):
        self._increase_running_job_cnt()
        job_inst.AddPostFinishCallback(lambda _: self._decrease_running_job_cnt())
        oneflow_api.LaunchJob(job_inst)

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
        loop = asyncio.get_event_loop()
        output_futures = {}
        for (
            output_name,
            pull_job_name,
        ) in self.inter_user_job_info_.output_or_var_op_name2pull_job_name.items():
            future = loop.create_future()
            pull_fn = self._make_pull_job_cb(output_name, future)
            pull_job_inst = job_instance_util.MakePullJobInstance(
                pull_job_name, output_name, pull_fn
            )
            self._run_job(pull_job_inst)
            output_futures[output_name] = future

        return output_futures

    def _make_pull_job_cb(self, output_name, future):
        output_lbn, output_job_name = self._query_output_lbn(output_name)
        split_axis = c_api_util.JobBuildAndInferCtx_GetSplitAxisFromProducerView(
            output_job_name, output_lbn
        )
        loop = asyncio.get_event_loop()

        def pull_fn(ofblob):
            ndarray_lists = ofblob.CopyToNdarrayLists()
            assert len(ndarray_lists) == 1
            ndarray_list = ndarray_lists[0]
            if len(ndarray_list) == 1:
                loop.call_soon_threadsafe(future.set_result, ndarray_list[0])
            else:
                assert split_axis is not None
                pull_result = np.concatenate(ndarray_list, axis=split_axis)
                loop.call_soon_threadsafe(future.set_result, pull_result)

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
