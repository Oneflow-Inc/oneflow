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
import oneflow.core.job.saved_model_pb2 as saved_model_pb
import oneflow.core.common.error_pb2 as error_util

from google.protobuf import text_format
from oneflow.python.framework.job_build_and_infer_error import JobBuildAndInferError


def api_load_model(load_model_proto):
    print("call oneflow_api LoadModel")
    assert type(load_model_proto) is saved_model_pb.LoadModelProto
    load_model_proto_str = text_format.MessageToString(load_model_proto)
    error_str = flow.oneflow_api.LoadModel(load_model_proto_str)

    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def load_alexnet():
    load_model_proto = saved_model_pb.LoadModelProto()
    load_model_proto.model_path = "saved_models"
    load_model_proto.version = 1

    load_model_proto.job_conf.job_name = "alexnet_eval_job"
    load_model_proto.job_conf.predict_conf.SetInParent()
    load_model_proto.parallel_conf.device_name.append("0:0")
    load_model_proto.parallel_conf.device_tag = "gpu"

    api_load_model(load_model_proto)


if __name__ == "__main__":
    load_alexnet()
