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
import oneflow.core.serving.prediction_service_pb2_grpc as prediction_service_grpc
import oneflow.core.serving.predict_pb2 as predict_pb
import oneflow.core.serving.model_spec_pb2 as model_spec_pb
import oneflow.core.serving.tensor_pb2 as tensor_pb
import oneflow.core.common.data_type_pb2 as data_type_pb
import oneflow.core.record.record_pb2 as record_pb
import grpc
import argparse
import cv2
import os
import time
import struct
import numpy as np


class PredictionServiceClient(object):
    def __init__(self):
        self.host = args.server_address
        self.server_port = args.server_port
        # instantiate a channel
        self.channel = grpc.insecure_channel(
            "{}:{}".format(self.host, self.server_port)
        )
        # bind the client and the server
        self.stub = prediction_service_grpc.PredictionServiceStub(self.channel)

    # TODO: sepcify ouputs
    def predict(self, model_name, function_name, signature_name, **kwargs):
        request = predict_pb.PredictRequest()

        request.model_spec.model_name = model_name
        request.model_spec.function_name = function_name
        request.model_spec.signature_name = signature_name

        for k, v in kwargs.items():
            if not isinstance(v, np.ndarray):
                v = np.array(v)

            input_tensor_proto = request.inputs[k]
            flow.serving.array_to_tensor_proto(v, input_tensor_proto)

        return self.stub.Predict(request)


def load_data_from_ofrecord(num_batchs, batch_size, ofrecord_data_file):
    image_list = []
    label_list = []

    with open(ofrecord_data_file, "rb") as reader:
        for i in range(num_batchs):
            images = []
            labels = []
            num_read = batch_size
            while num_read > 0:
                record_head = reader.read(8)
                if record_head is None or len(record_head) != 8:
                    break

                ofrecord = record_pb.OFRecord()
                ofrecord_byte_size = struct.unpack("q", record_head)[0]
                ofrecord.ParseFromString(reader.read(ofrecord_byte_size))

                image_raw_bytes = ofrecord.feature["encoded"].bytes_list.value[0]
                image = cv2.imdecode(
                    np.frombuffer(image_raw_bytes, np.uint8), cv2.IMREAD_COLOR
                ).astype(np.float32)
                images.append(image)
                label = ofrecord.feature["class/label"].int32_list.value[0]
                labels.append(label)
                num_read -= 1

            if num_read == 0:
                image_list.append(np.stack(images))
                label_list.append(np.array(labels, dtype=np.int32))
            else:
                break

    return image_list, label_list


def get_parser():
    parser = argparse.ArgumentParser("flags for grpc client demo")
    parser.add_argument("--server_address", type=str, default="localhost", help="")
    parser.add_argument("--server_port", type=int, default=9887, help="")
    parser.add_argument(
        "--test_images_path",
        type=str,
        default="/dataset/http_service_demo_client_images/",
        help="",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # test_code
    if not os.path.exists(args.test_images_path):
        raise Exception("Images path does not existed.")

    client = PredictionServiceClient()
    images = os.listdir(args.test_images_path)

    num_batchs = 10
    batch_size = 4
    ofrecord_data_path = os.path.join("/dataset/imagenet_227/train/32/", "part-0")
    image_list, label_list = load_data_from_ofrecord(
        num_batchs, batch_size, ofrecord_data_path
    )
    for i in range(num_batchs):
        image = image_list[i]
        label = label_list[i]

        print("#### iter{} ####".format(i))
        print("send image to server")
        response = client.predict(
            "alexnet", "alexnet_inference", "regress", image=image, label=label
        )
        print("get result from server:")
        for output_name, tensor_proto in response.outputs.items():
            arr = flow.serving.tensor_proto_to_array(tensor_proto)
            print(
                "{} shape: {}, dtype: {}, data:\n{}".format(
                    output_name, arr.shape, arr.dtype, arr
                )
            )

        time.sleep(2)
