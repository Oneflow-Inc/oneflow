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
import oneflow as flow
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_ctx
import oneflow.python.framework.session_util as session_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.scope_util as scope_util
import oneflow.python.framework.runtime_mode as runtime_mode
import oneflow.core.job.saved_model_pb2 as saved_model_pb
import oneflow.core.job.job_conf_pb2 as job_conf_pb
import oneflow.core.common.error_pb2 as error_util

from google.protobuf import text_format
from contextlib import contextmanager
from oneflow.python.framework.job_build_and_infer_error import JobBuildAndInferError


class InferenceSession(object):
    def __init__(self, config_proto=None):
        self.is_mirrored_ = False
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
        c_api_util.StopLazyGlobalSession()
        c_api_util.DestroyLazyGlobalSession()
        c_api_util.DestroyEnv()

    def __del__(self):
        self.close()

    @contextmanager
    def open(self, job_name):
        # setup job
        job_conf = job_conf_pb.JobConfigProto()
        job_conf.job_name = job_name
        c_api_util.JobBuildAndInferCtx_Open(job_name)
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
            print("\top_name:", op_conf.name)
            compile_ctx.CurJobAddOp(op_conf)

        c_api_util.CurJobBuildAndInferCtx_Complete()

    def launch(self):
        c_api_util.StartLazyGlobalSession()
        self.inter_user_job_info_ = c_api_util.GetInterUserJobInfo()
        print("inter_user_job_info:\n{}".format(self.inter_user_job_info_))


def load_saved_model(model_meta_file_path):
    saved_model = saved_model_pb.SavedModel()
    with open(model_meta_file_path, "rb") as f:
        text_format.Merge(f.read(), saved_model)
    # print("saved model proto:", "\n", saved_model)
    return saved_model


def load_model():
    flow.env.init()

    load_model_proto = saved_model_pb.LoadModelProto()
    load_model_proto.model_path = "saved_models"
    load_model_proto.version = 1

    load_model_proto.job_conf.job_name = "alexnet_eval_job"
    load_model_proto.job_conf.predict_conf.SetInParent()
    load_model_proto.parallel_conf.device_name.append("0:0")
    load_model_proto.parallel_conf.device_tag = "gpu"

    saved_model_path = "saved_models"
    version = 1
    model_meta_file_path = os.path.join(
        saved_model_path, str(version), "saved_model.prototxt"
    )
    saved_model_proto = load_saved_model(model_meta_file_path)

    sess = InferenceSession()
    for job_name, net in saved_model_proto.graphs.items():
        print("job:", job_name)
        with sess.open(job_name) as sess:
            sess.compile(net.op)

    job_set = c_api_util.GetJobSet()
    for job in job_set.job:
        print(job.job_conf.job_name)
        for op_conf in job.net.op:
            print("\t", op_conf.name)

    sess.launch()
    time.sleep(30)


if __name__ == "__main__":
    load_model()
