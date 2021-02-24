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
import asyncio
import contextlib
import inspect
import numpy as np
import os
import enum
import google.protobuf.text_format as text_format

import oneflow as flow
import oneflow_api
import oneflow_api.oneflow.core.job.job_conf as job_conf_proto_cfg

import oneflow_api.oneflow.core.operator.interface_blob_conf as interface_blob_conf_proto_cfg
import oneflow_api.oneflow.core.common.shape as shape_proto_cfg
import oneflow_api.oneflow.core.common.data_type as dtype_proto_cfg
import oneflow.core.job.job_conf_pb2 as job_conf_proto
import oneflow.core.operator.interface_blob_conf_pb2 as interface_blob_conf_proto
import oneflow.core.serving.saved_model_pb2 as saved_model_pb
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


def _is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True


def _find_model_latest_version(saved_model_dir):
    version_dirs = []
    for f in os.listdir(saved_model_dir):
        if os.path.isdir(os.path.join(saved_model_dir, f)) and _is_int(f):
            version_dirs.append(f)

    version_dirs.sort(reverse=True, key=lambda x: int(x))
    return version_dirs[0]


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
    assert isinstance(
        inferface_blob_conf_proto, interface_blob_conf_proto.InterfaceBlobConf
    )
    assert isinstance(
        mut_inferface_blob_conf_cfg, interface_blob_conf_proto_cfg.InterfaceBlobConf
    )

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

    mut_inferface_blob_conf_cfg.set_is_dynamic(inferface_blob_conf_proto.is_dynamic)
    mut_inferface_blob_conf_cfg.set_is_tensor_list(
        inferface_blob_conf_proto.is_tensor_list
    )


@oneflow_export("serving.ModelVersionPolicy")
class ModelVersionPolicy(enum.Enum):
    LATEST = 1


@oneflow_export("serving.SessionOption")
class SessionOption(object):
    def __init__(self):
        self.device_tag = "gpu"
        self.device_num = 1
        self.is_mirrored_view = False


@oneflow_export("serving.InferenceSession")
class InferenceSession(object):
    class SessionStatus(enum.Enum):
        OPEN = 1
        RUNNING = 2
        CLOSED = 3

    def __init__(self, option=None):
        if option is None:
            self.option_ = SessionOption()
        else:
            assert isinstance(option, SessionOption)
            self.option_ = option

        self.is_mirrored_ = self.option_.is_mirrored_view
        self.checkpoint_path_ = None
        self.config_proto_ = None
        self.job_name2job_conf_ = {}
        self.inter_user_job_info_ = None
        self.cur_job_name_ = None
        self.inferface_name2info_ = {}
        self.output_name2future_ = {}
        self.job_futures_ = []
        self.status_ = None

        self._init_event_loop()
        self.init()

    def __del__(self):
        if self.status_ != self.SessionStatus.CLOSED:
            self.close()

    def _init_event_loop(self):
        self.event_loop_ = asyncio.get_event_loop()
        if self.event_loop_.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.event_loop_ = asyncio.get_event_loop()

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

    def close(self):
        self.event_loop_.run_until_complete(self.wait_for_all_jobs_finished())
        self.event_loop_.close()

        if self.status_ == self.SessionStatus.RUNNING:
            oneflow_api.StopLazyGlobalSession()
            oneflow_api.DestroyLazyGlobalSession()
        elif self.status_ == self.SessionStatus.OPEN:
            oneflow_api.DestroyLazyGlobalSession()
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

    def set_checkpoint_path(self, checkpoint_path):
        self._check_status(self.SessionStatus.OPEN)
        self.checkpoint_path_ = checkpoint_path

    def set_job_signature(self, job_name, signature):
        assert isinstance(signature, job_conf_proto.JobSignatureDef)
        job_conf = self._get_job_conf(job_name)
        _signature_proto_to_cfg(signature, job_conf.mutable_signature())

    def set_job_batch_size(self, job_name, batch_size):
        self._check_status(self.SessionStatus.OPEN)
        job_conf = self._get_job_conf(job_name)
        for _, mut_input_def in job_conf.mutable_signature().mutable_inputs().items():
            mut_shape = mut_input_def.mutable_blob_conf().mutable_shape()
            mut_shape.mutable_dim()[0] = batch_size

    def _get_job_conf(self, job_name):
        if job_name in self.job_name2job_conf_:
            return self.job_name2job_conf_[job_name]
        else:
            job_conf = job_conf_proto_cfg.JobConfigProto()
            job_conf.set_job_name(job_name)
            self.job_name2job_conf_[job_name] = job_conf
            return job_conf

    @contextlib.contextmanager
    def open(self, job_name, signature=None, batch_size=None):
        self._check_status(self.SessionStatus.OPEN)
        c_api_util.JobBuildAndInferCtx_Open(job_name)

        if signature is not None:
            self.set_job_signature(job_name, signature)

        if isinstance(batch_size, int):
            self.set_job_batch_size(job_name, batch_size)

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
                self.cur_job_name_ = job_name
                yield self
                self.cur_job_name_ = None

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
        oneflow_api.CurJobBuildAndInferCtx_Rebuild()

    def launch(self):
        self._check_status(self.SessionStatus.OPEN)
        oneflow_api.StartLazyGlobalSession()
        self.inter_user_job_info_ = c_api_util.GetInterUserJobInfo()
        self._run_load_checkpoint_job()
        self.status_ = self.SessionStatus.RUNNING

    def load_saved_model(
        self,
        saved_model_dir,
        model_version=ModelVersionPolicy.LATEST,
        saved_model_meta_file_basename="saved_model",
        graph_name=None,
        signature_name=None,
    ):
        if not os.path.isdir(saved_model_dir):
            raise ValueError("{} is not a valid directory".format(saved_model_dir))

        if isinstance(model_version, int):
            pass
        elif model_version == ModelVersionPolicy.LATEST:
            model_version = _find_model_latest_version(saved_model_dir)
        else:
            raise NotImplementedError

        saved_model_path = os.path.join(saved_model_dir, str(model_version))
        if not os.path.isdir(saved_model_path):
            raise ValueError(
                "version {} of saved model in dir {} do not exist".format(
                    model_version, saved_model_dir
                )
            )

        subfiles = list(os.listdir(saved_model_path))
        saved_model_meta_pb_filename = saved_model_meta_file_basename + ".pb"
        saved_model_meta_prototxt_filename = (
            saved_model_meta_file_basename + ".prototxt"
        )
        saved_model_proto = saved_model_pb.SavedModel()
        if saved_model_meta_pb_filename in subfiles:
            saved_model_meta_file_path = os.path.join(
                saved_model_path, saved_model_meta_pb_filename
            )
            with open(saved_model_meta_file_path, "rb") as f:
                saved_model_proto.ParseFromString(f.read())
        elif saved_model_meta_prototxt_filename in subfiles:
            saved_model_meta_file_path = os.path.join(
                saved_model_path, saved_model_meta_prototxt_filename
            )
            with open(saved_model_meta_file_path, "rt") as f:
                text_format.Merge(f.read(), saved_model_proto)
        else:
            raise ValueError(
                "saved model meta file {} do not exist in {}".format(
                    saved_model_meta_file_basename, saved_model_path
                )
            )
        # set checkpoint
        self.set_checkpoint_path(
            os.path.join(saved_model_path, saved_model_proto.checkpoint_dir)
        )
        # get signature
        signature = None
        if graph_name is None:
            graph_name = saved_model_proto.default_graph_name
        else:
            if graph_name not in saved_model_proto.graphs:
                raise ValueError("graph {} do not exist".format(graph_name))
        graph_def = saved_model_proto.graphs[graph_name]
        if signature_name is None and graph_def.HasField("default_signature_name"):
            signature_name = graph_def.default_signature_name
        if signature_name is not None:
            if signature_name not in graph_def.signatures:
                raise ValueError("signature {} do not exist".format(signature_name))
            else:
                signature = graph_def.signatures[signature_name]

        # compile job
        with self.open(graph_name, signature):
            self.compile(graph_def.op_list)

    def print_job_set(self):
        self._check_status(self.SessionStatus.OPEN, self.SessionStatus.RUNNING)
        job_set = c_api_util.GetJobSet()
        for job in job_set.job:
            print("job_name:", job.job_conf.job_name)
            for op_conf in job.net.op:
                print("\top_name:", op_conf.name)

    def list_jobs(self):
        self._check_status(self.SessionStatus.RUNNING)
        return list(self.job_name2job_conf_.keys())

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

    def input_info(self, input_name, job_name=None):
        return self._get_op_blob_info(job_name, input_name, "out")

    def output_info(self, output_name, job_name=None):
        return self._get_op_blob_info(job_name, output_name, "in")

    def _get_op_blob_info(self, job_name, op_name, blob_name):
        self._check_status(self.SessionStatus.OPEN, self.SessionStatus.RUNNING)
        if op_name in self.inferface_name2info_:
            return self.inferface_name2info_[op_name]

        job_name = job_name or self.cur_job_name_
        if job_name is None:
            raise ValueError("please specify job_name")

        lbn = oneflow_api.JobBuildAndInferCtx_GetOpBlobLbn(job_name, op_name, blob_name)
        shape = c_api_util.JobBuildAndInferCtx_GetStaticShape(job_name, lbn)
        dtype = c_api_util.JobBuildAndInferCtx_GetDataType(job_name, lbn)
        dtype = dtype_util.convert_proto_dtype_to_oneflow_dtype(dtype)
        # TODO: other info
        info = dict(shape=shape, dtype=dtype)
        self.inferface_name2info_[op_name] = info
        return info

    def run(self, job_name, **kwargs):
        self._check_status(self.SessionStatus.RUNNING)
        return self.event_loop_.run_until_complete(self.async_run(job_name, **kwargs))

    async def async_run(self, job_name, **kwargs):
        self._check_status(self.SessionStatus.RUNNING)
        self._run_push_jobs(**kwargs)
        job_inst = job_instance_util.MakeUserJobInstance(job_name)
        self._run_job(job_inst)
        output_futures = tuple(self._run_pull_jobs(job_name).values())
        return await asyncio.gather(*output_futures)

    def _run_job(self, job_inst):
        future = self.event_loop_.create_future()

        def job_finish_cb(_):
            self.event_loop_.call_soon_threadsafe(future.set_result, None)

        job_inst.AddPostFinishCallback(job_finish_cb)
        oneflow_api.LaunchJob(job_inst)
        self.job_futures_.append(future)

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

    def _run_pull_jobs(self, user_job_name):
        output_futures = {}
        for (
            output_name,
            pull_job_name,
        ) in self.inter_user_job_info_.output_or_var_op_name2pull_job_name.items():
            future = self.event_loop_.create_future()
            pull_fn = self._make_pull_job_cb(output_name, user_job_name, future)
            pull_job_inst = job_instance_util.MakePullJobInstance(
                pull_job_name, output_name, pull_fn
            )
            self._run_job(pull_job_inst)
            output_futures[output_name] = future

        return output_futures

    def _make_pull_job_cb(self, output_name, user_job_name, future):
        output_lbn = oneflow_api.JobBuildAndInferCtx_GetOpBlobLbn(
            user_job_name, output_name, "out"
        )
        split_axis = c_api_util.JobBuildAndInferCtx_GetSplitAxisFromProducerView(
            user_job_name, output_lbn
        )

        def pull_fn(ofblob):
            ndarray_lists = ofblob.CopyToNdarrayLists()
            assert len(ndarray_lists) == 1
            ndarray_list = ndarray_lists[0]
            if len(ndarray_list) == 1:
                self.event_loop_.call_soon_threadsafe(
                    future.set_result, ndarray_list[0]
                )
            else:
                assert split_axis is not None
                pull_result = np.concatenate(ndarray_list, axis=split_axis)
                self.event_loop_.call_soon_threadsafe(future.set_result, pull_result)

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
        )
        self._run_job(load_checkpoint_job_inst)

    async def wait_for_all_jobs_finished(self):
        await asyncio.gather(*self.job_futures_)
        self.job_futures_ = []
