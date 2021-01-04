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
import grpc
import numpy as np
import oneflow as flow
import oneflow.core.serving.prediction_service_pb2_grpc as prediction_service_grpc
import oneflow.core.serving.predict_pb2 as predict_pb

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("serving.PredictionGrpcClient")
class PredictionGrpcClient(object):
    def __init__(self, server_address, server_port):
        # instantiate a channel
        channel = grpc.insecure_channel("{}:{}".format(server_address, server_port))
        # bind the client and the server
        self.stub_ = prediction_service_grpc.PredictionServiceStub(channel)

    # TODO: sepcify ouputs
    def predict(self, model_name, graph_name, **kwargs):
        request = predict_pb.PredictRequest()

        request.model_spec.model_name = model_name
        if isinstance(graph_name, str):
            request.model_spec.graph_name = graph_name

        for k, v in kwargs.items():
            if not isinstance(v, np.ndarray):
                v = np.array(v)

            input_tensor_proto = request.inputs[k]
            flow.serving.array_to_tensor_proto(v, input_tensor_proto)

        return self.stub_.Predict(request)
