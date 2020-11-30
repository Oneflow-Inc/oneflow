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

from oneflow.python.oneflow_export import oneflow_export


# TODO list:
# -- support multi model serving: place loaded model and related session in independent subprocess
# -- support more prediction service and model service (get model meta or status, reload model)
# -- support stream request and stream response for big-size inputs and outputs
# -- version policy of model
# -- aiohttp server
# -- batching
# -- hot update model, two ways: 1) watch model config file changing and reload;
#    2) through ReloadConfig request

# TODO: consider big tensor using stream request and stream response
# because of the limit "Received message larger than max (4946900 vs. 4194304)"
class PredictionService(prediction_grpc.PredictionServiceServicer):
    def __init__(self, model_server):
        self.model_server_ = model_server

    async def Predict(self, predict_request, context):
        print("accept predict request")
        print("model name:", predict_request.model_spec.model_name)
        for input_name, tensor in predict_request.inputs.items():
            print(
                "input {} shape: {}, dtype: {}".format(
                    input_name, tuple(tensor.shape.dim), tensor.data_type
                )
            )

        outputs = await self.model_server_.predict(predict_request)
        print("predict request processing finished")

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
        self.model_name2default_graph_name_ = {}

    def build_and_start(self):
        # read model config
        # TODO: support pass model config directly
        assert server_config.HasField("model_config_prototxt")
        model_config_list_proto = model_config_pb.ModelConfigList()
        with open(server_config.model_config_prototxt, "rb") as f:
            text_format.Merge(f.read(), model_config_list_proto)

        # load model
        # TODO: support multi model through subprocess
        assert len(model_config_list_proto.model_config) == 1
        model_config = model_config_list_proto.model_config[0]
        self.load_model(model_config)

        # start grpc server
        asyncio.run(self.build_and_start_grpc_server())
        print("server started")

    def load_model(self, model_config):
        model_name = model_config.name
        if model_name not in self.model_name2versions2session_:
            self.model_name2versions2session_[model_name] = {}

        model_path = model_config.saved_model_path
        model_version = find_model_latest_version(model_path)
        if model_version in self.model_name2versions2session_[model_name]:
            raise ValueError(
                "The version ({}) of model ({}) already exist".format(
                    model_name, model_version
                )
            )

        # load saved_model proto
        saved_model_proto = saved_model_pb.SavedModel()
        if model_config.HasField("saved_model_pb_file_name"):
            pb_file_path = os.path.join(
                model_path, str(model_version), model_config.saved_model_pb_file_name
            )
            with open(pb_file_path, "rb") as f:
                saved_model_proto.ParseFromString(f.read())
        elif model_config.HasField("saved_model_prototxt_file_name"):
            prototxt_file_path = os.path.join(
                model_path,
                str(model_version),
                model_config.saved_model_prototxt_file_name,
            )
            with open(prototxt_file_path, "rt") as f:
                text_format.Merge(f.read(), saved_model_proto)
        else:
            raise ValueError("")

        # create session
        option = flow.SessionOption()
        if model_config.HasField("device_tag"):
            option.device_tag = model_config.device_tag
        if model_config.HasField("device_num"):
            option.device_num = model_config.device_num
        sess = flow.SimpleSession(option)

        # set checkpoint dir
        checkpoint_path = os.path.join(
            model_path, str(model_version), saved_model_proto.checkpoint_dir,
        )
        sess.set_checkpoint_path(checkpoint_path)

        # add graph
        def add_graph(graph_name, signature_name=None):
            graph_def = saved_model_proto.graphs[graph_name]
            if signature_name is None:
                if graph_def.HasField("default_signature_name"):
                    signature_name = graph_def.default_signature_name
                else:
                    raise ValueError(
                        "graph ({graph_name}) of saved model {model_name} has no default signature name"
                        ", you should set signature_name for graph in model_config".format(
                            graph_name=graph_name, model_name=model_name,
                        )
                    )

            signature_def = graph_def.signatures[signature_name]
            sess.setup_job_signature(graph_name, signature_def)
            with sess.open(graph_name):
                sess.compile(graph_def.op_list)

            if model_name not in self.model_name2default_graph_name_:
                self.model_name2default_graph_name_[model_name] = graph_name

            print(
                "graph ({}) with signature ({}) is added to model ({})".format(
                    graph_name, signature_name, model_name
                )
            )

        graph_name = None
        for graph_config in model_config.graph_config:
            graph_name = graph_config.name
            signature_name = None
            if graph_config.HasField("signature_name"):
                signature_name = graph_config.signature_name
            add_graph(graph_name, signature_name)

        # these is no any graph_config in model_config
        if graph_name is None:
            if saved_model_proto.HasField("default_graph_name"):
                graph_name = saved_model_proto.default_graph_name
                add_graph(graph_name)
            else:
                raise ValueError(
                    "saved model {} has no default graph name, you should set model_config.graph_name".format(
                        model_name
                    )
                )

        sess.launch()

        self.model_name2versions2session_[model_name][model_version] = sess

        print("version {} of model {} loaded".format(model_version, model_name))
        input_names = sess.list_inputs()
        print("input names:", input_names)
        for input_name in input_names:
            print('input "{}" info: {}'.format(input_name, sess.input_info(input_name)))

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
        model_name = model_spec.model_name
        if model_spec.HasField("version"):
            model_version = model_spec.version
            if model_version not in self.model_name2versions2session_[model_name]:
                raise ValueError("model version {} not exist".format(model_version))
        else:
            versions = list(self.model_name2versions2session_[model_name].keys())
            model_version = versions[-1]

        sess = self.model_name2versions2session_[model_name][model_version]
        print("session got")

        input_names = sess.list_inputs()
        inputs = {}
        for input_name, tensor_proto in predict_request.inputs.items():
            if input_name in input_names:
                inputs[input_name] = flow.serving.tensor_proto_to_array(tensor_proto)
                print("{} dtype: {}".format(input_name, inputs[input_name].dtype))

        print("ready to run")
        graph_name = None
        if model_spec.HasField("graph_name"):
            graph_name = model_spec.graph_name
        elif model_name in self.model_name2default_graph_name_:
            graph_name = self.model_name2default_graph_name_[model_name]
        else:
            raise ValueError(
                "model ({}) has no default graph, graph_name must be set in model_spec".format(
                    model_name
                )
            )

        outputs = sess.run(graph_name, **inputs)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=9887, help="port for grpc",
    )
    parser.add_argument(
        "--model_config_prototxt",
        type=str,
        default="model_config.prototxt",
        help="model config file",
    )
    args = parser.parse_args()

    server_config = server_config_pb.ServerConfig()
    server_config.port = args.port
    server_config.model_config_prototxt = args.model_config_prototxt
    server = ModelServer(server_config)
    server.build_and_start()
