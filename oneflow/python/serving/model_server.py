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
import oneflow.core.serving.prediction_service_pb2_grpc as prediction_grpc
import oneflow.core.serving.predict_pb2 as predict_pb
import oneflow.core.serving.saved_model_pb2 as saved_model_pb
import oneflow.core.serving.model_config_pb2 as model_config_pb
import oneflow.core.serving.server_config_pb2 as server_config_pb
import google.protobuf.text_format as text_format
import grpc
import os
import asyncio
import logging

from oneflow.python.oneflow_export import oneflow_export


# TODO list:
# -- support multi model serving: place loaded model and related session in independent subprocess
# -- support more prediction service and model service (get model meta or status, reload model)
# -- support stream request and stream response for big-size inputs and outputs
# -- version policy of model
# -- aiohttp server
# -- complete async server
# -- fix proto
# -- update and change SimpleSession interface
# -- batching util
# -- hot update model, two ways: 1) watch model config file changing and reload;
#    2) through ReloadConfig request

# TODO: consider big tensor using stream request and stream response
# because of the limit "Received message larger than max (4946900 vs. 4194304)"
class PredictionService(prediction_grpc.PredictionServiceServicer):
    def __init__(self, model_server):
        self.model_server_ = model_server

    async def Predict(self, predict_request, context):
        print("accept request")
        print("model name:", predict_request.model_spec.model_name)
        print("function name:", predict_request.model_spec.function_name)
        print("signature name:", predict_request.model_spec.signature_name)
        for input_name, tensor in predict_request.inputs.items():
            print(
                "input {} shape: {}, dtype: {}".format(
                    input_name, tuple(tensor.shape.dim), tensor.data_type
                )
            )

        outputs = await self.model_server_.predict(predict_request)
        print("predict finished")
        predict_response = predict_pb.PredictResponse()
        for output_name, output in outputs.items():
            output_tensor_proto = predict_response.outputs[output_name]
            flow.serving.array_to_tensor_proto(output, output_tensor_proto)

        return predict_response


# TODO: sperate model_server and base_server (process request and dispatch)
@oneflow_export("serving.Server")
class ModelServer(object):
    def __init__(self, server_config):
        self.server_config_ = server_config
        self.model_name2versions2session_ = {}

    def build_and_start(self):
        # read model config
        # TODO: support pass model config directly
        assert server_config.HasField("model_config_file")
        self.model_config_ = model_config_pb.ModelServerConfig()
        with open(server_config.model_config_file, "rb") as f:
            text_format.Merge(f.read(), self.model_config_)

        # load model
        # TODO: support multi model through subprocess
        assert len(self.model_config_.model_config_list) == 1
        model_config = self.model_config_.model_config_list[0]
        self.load_model(model_config)
        print("model loaded")

        # start grpc server
        asyncio.run(self.build_and_start_grpc_server())
        print("server started")

    def load_model(self, model_config):
        latest_model_version = find_model_latest_version(model_config.path)
        model_meta_file_path = os.path.join(
            model_config.path, str(latest_model_version), "saved_model.prototxt"
        )
        saved_model_proto = load_saved_model(model_meta_file_path)
        sess = flow.SimpleSession()
        checkpoint_path = os.path.join(
            model_config.path,
            str(latest_model_version),
            saved_model_proto.checkpoint_dir[0],
        )
        sess.set_checkpoint_path(checkpoint_path)
        for job_name, signature in saved_model_proto.signatures_v2.items():
            sess.setup_job_signature(job_name, signature)

        for job_name, net in saved_model_proto.graphs.items():
            with sess.open(job_name) as sess:
                sess.compile(net.op)

        sess.launch()
        if model_config.name not in self.model_name2versions2session_:
            self.model_name2versions2session_[model_config.name] = {}
        self.model_name2versions2session_[model_config.name][
            latest_model_version
        ] = sess

    async def build_and_start_grpc_server(self):
        grpc_server = grpc.aio.server()
        prediction_grpc.add_PredictionServiceServicer_to_server(
            PredictionService(self), grpc_server
        )
        listen_addr = "0.0.0.0:{}".format(self.server_config_.port)
        grpc_server.add_insecure_port(listen_addr)
        await grpc_server.start()
        print("grpc server started, listen at {}".format(listen_addr))
        await grpc_server.wait_for_termination()

    # TODO: support batching
    async def predict(self, predict_request):
        print("predict")
        model_spec = predict_request.model_spec
        if model_spec.HasField("version"):
            model_version = model_spec.version
            assert model_version in self.model_name2versions2session_[model_config.name]
        else:
            versions = list(self.model_name2versions2session_[model_config.name].keys())
            model_version = versions[-1]
        sess = self.model_name2versions2session_[model_config.name][model_version]
        print("session got")

        input_names = sess.list_inputs()
        inputs = {}
        for input_name, tensor_proto in predict_request.inputs.items():
            if input_name in input_names:
                inputs[input_name] = flow.serving.tensor_proto_to_array(tensor_proto)
                print("{} dtype: {}".format(input_name, inputs[input_name].dtype))

        print("ready to run")
        outputs = sess.run(model_spec.function_name, **inputs)
        output_dict = {}
        for i, output_name in enumerate(sess.list_outputs()):
            output_dict[output_name] = outputs[i]

        return output_dict


def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True


def find_model_latest_version(saved_model_path):
    version_dirs = []
    for f in os.listdir(saved_model_path):
        if os.path.isdir(os.path.join(saved_model_path, f)) and is_int(f):
            version_dirs.append(f)

    version_dirs.sort(reverse=True, key=lambda x: int(x))
    return version_dirs[0]


def load_saved_model(model_meta_file_path):
    saved_model_proto = saved_model_pb.SavedModel()
    with open(model_meta_file_path, "rb") as f:
        text_format.Merge(f.read(), saved_model_proto)
    return saved_model_proto


if __name__ == "__main__":
    print("start model server")
    # test_code: generate model_config
    model_server_config = model_config_pb.ModelServerConfig()
    model_config = model_config_pb.ModelConfig()
    model_config.name = "alexnet"
    model_config.path = "alexnet_models"
    model_server_config.model_config_list.append(model_config)
    with open("model_config.prototxt", "w") as f:
        f.write("{}\n".format(str(model_server_config)))

    # test_code: init server
    server_config = server_config_pb.ServerConfig()
    server_config.port = 9887
    server_config.model_config_file = "model_config.prototxt"
    server = ModelServer(server_config)
    server.build_and_start()
