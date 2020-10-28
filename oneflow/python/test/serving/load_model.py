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
import os
import time
import threading
import oneflow as flow
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_ctx
import oneflow.python.framework.session_util as session_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.scope_util as scope_util
import oneflow.python.framework.runtime_mode as runtime_mode
import oneflow.python.framework.job_instance as job_instance_util
import oneflow.core.job.saved_model_pb2 as saved_model_pb
import oneflow.core.job.job_conf_pb2 as job_conf_pb

from google.protobuf import text_format
from contextlib import contextmanager
from oneflow.python.framework.job_build_and_infer_error import JobBuildAndInferError


class InferenceSession(object):
    def __init__(self, config_proto=None):
        self.job_name2job_conf_ = {}
        self.is_mirrored_ = False
        self.cond_var_ = threading.Condition()
        self.running_job_cnt_ = 0
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

    def close(self):
        self._synchronize()
        c_api_util.StopLazyGlobalSession()
        c_api_util.DestroyLazyGlobalSession()
        c_api_util.DestroyEnv()

    def __del__(self):
        self.close()

    def setup_job_conf(self, job_name, signature):
        job_conf = job_conf_pb.JobConfigProto()
        job_conf.job_name = job_name
        job_conf.signature.CopyFrom(signature)
        self.job_name2job_conf_[job_name] = job_conf

    def get_job_conf(self, job_name):
        if job_name in self.job_name2job_conf_:
            return self.job_name2job_conf_[job_name]
        else:
            job_conf = job_conf_pb.JobConfigProto()
            job_conf.job_name = job_name
            self.job_name2job_conf_[job_name] = job_conf
            return job_conf

    @contextmanager
    def open(self, job_name):
        c_api_util.JobBuildAndInferCtx_Open(job_name)
        job_conf = self.get_job_conf(job_name)
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
        for op_conf in op_list:
            compile_ctx.CurJobAddOp(op_conf)

        c_api_util.CurJobBuildAndInferCtx_Complete()

    def launch(self):
        c_api_util.StartLazyGlobalSession()
        self.inter_user_job_info_ = c_api_util.GetInterUserJobInfo()
        print("inter_user_job_info:\n{}".format(self.inter_user_job_info_))
        # TODO: load checkpoint and setup model init job

    def run(self, job_name):
        job_inst = job_instance_util.MakeUserJobInstance(job_name)
        self._increase_running_job_cnt()
        job_inst.AddPostFinishCallback(lambda _: self._decrease_running_job_cnt())
        c_api_util.LaunchJob(job_inst)

    def _synchronize(self):
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


def load_saved_model(model_meta_file_path):
    saved_model = saved_model_pb.SavedModel()
    with open(model_meta_file_path, "rb") as f:
        text_format.Merge(f.read(), saved_model)
    # print("saved model proto:", "\n", saved_model)
    return saved_model


def load_model():
    saved_model_path = "saved_models_v2"
    version = 1
    model_meta_file_path = os.path.join(
        saved_model_path, str(version), "saved_model.prototxt"
    )
    saved_model_proto = load_saved_model(model_meta_file_path)

    sess = InferenceSession()

    for job_name, signature in saved_model_proto.signatures_v2.items():
        sess.setup_job_conf(job_name, signature)

    for job_name, net in saved_model_proto.graphs.items():
        with sess.open(job_name) as sess:
            sess.compile(net.op)

    job_set = c_api_util.GetJobSet()
    for job in job_set.job:
        print(job.job_conf.job_name)
        for op_conf in job.net.op:
            print("\t", op_conf.name)

    sess.launch()
    sess.run("alexnet_eval_job")
    # time.sleep(10)


if __name__ == "__main__":
    load_model()
