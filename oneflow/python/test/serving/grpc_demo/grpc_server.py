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
import oneflow.core.serving.saved_model_pb2 as saved_model_pb

import argparse
import cv2
import google.protobuf.text_format as text_format
import grpc
import numpy as np
import os
import prediction_service_pb2_grpc as grpc_service_pb2
import prediction_service_pb2 as predict_message_pb2

from concurrent import futures
from oneflow.python.test.serving.imagenet1000_clsidx_to_labels import clsidx_2_labels


def get_parser():
    def float_list(x):
        return list(map(float, x.split(",")))

    parser = argparse.ArgumentParser("flags for grpc server demo")
    parser.add_argument("--server_address", type=str, default="localhost", help="")
    parser.add_argument("--server_port", type=int, default=8000, help="")
    parser.add_argument(
        "--saved_model_path", type=str, default="./resnet50_models", help=""
    )
    parser.add_argument("--model_version", type=int, default=1, help="")
    parser.add_argument(
        "--rgb-mean",
        type=float_list,
        default=[123.68, 116.779, 103.939],
        help="a tuple of size 3 for the mean rgb",
    )
    parser.add_argument(
        "--rgb-std",
        type=float_list,
        default=[58.393, 57.12, 57.375],
        help="a tuple of size 3 for the std rgb",
    )
    return parser


parser = get_parser()
args = parser.parse_args()


def preprocess_image(im, height, width):
    im = cv2.resize(im.astype("uint8"), (height, width))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype("float32")
    im = (im - args.rgb_mean) / args.rgb_std
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, "float32")


def load_saved_model(model_meta_file_path):
    saved_model_proto = saved_model_pb.SavedModel()
    with open(model_meta_file_path, "rb") as f:
        text_format.Merge(f.read(), saved_model_proto)
    return saved_model_proto


print("load saved resnet 50 model.")
flow.clear_default_session()
model_meta_file_path = os.path.join(
    args.saved_model_path, str(args.model_version), "saved_model.prototxt"
)
saved_model_proto = load_saved_model(model_meta_file_path)
infer_session = flow.SimpleSession()
checkpoint_path = os.path.join(
    args.saved_model_path, str(args.model_version), saved_model_proto.checkpoint_dir[0]
)
infer_session.set_checkpoint_path(checkpoint_path)
for job_name, signature in saved_model_proto.signatures_v2.items():
    # print(signature)
    infer_session.setup_job_signature(job_name, signature)

for job_name, net in saved_model_proto.graphs.items():
    with infer_session.open(job_name) as infer_session:
        infer_session.compile(net.op)

# sess.print_job_set()
infer_session.launch()
input_names = infer_session.list_inputs()
print("input names:", input_names)
for input_name in input_names:
    print(
        'input "{}" info: {}'.format(input_name, infer_session.input_info(input_name))
    )

batch_size, channel, height, width = infer_session.input_info(input_names[0])["shape"]


class PredictionServicer(grpc_service_pb2.PredictionService):
    def __init__(self, *args, **kwargs):
        print("create servicer")

    def Classify(self, classification_request, context):
        message_from_client = f'Server received input image size: "{len(classification_request.numpy_inputs.numpy_arrays)}"'
        print(message_from_client)

        input_array = classification_request.numpy_inputs.numpy_arrays[0]

        deserialized_bytes = np.frombuffer(input_array.np_array_content, dtype=np.uint8)
        image = np.reshape(
            deserialized_bytes, newshape=tuple(input_array.np_array_shapes)
        )

        image = preprocess_image(image, height, width)
        images = np.repeat(image, batch_size, axis=0).astype(np.float32)

        predictions = infer_session.run("resnet_inference", image=images)

        clsidxs = np.argmax(predictions[0], axis=1)
        probs = np.max(predictions[0], axis=1)

        print(
            "predicted class name: %s, prob: %f\n"
            % (clsidx_2_labels[clsidxs[0]], probs[0])
        )

        respond = predict_message_pb2.ClassificationResponse()
        class_lists = predict_message_pb2.ClassLists()
        result = {
            "predicted_label": clsidx_2_labels[clsidxs[0]],
            "predicted_score": probs[0],
        }
        class_lists.classes.append(predict_message_pb2.SingleClass(**result))
        respond.result.classlists.append(class_lists)
        return respond


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpc_service_pb2.add_PredictionServiceServicer_to_server(
        PredictionServicer(), server
    )
    server.add_insecure_port("%s:%d" % (args.server_address, args.server_port))
    server.start()
    print("start gprc server")
    server.wait_for_termination()


serve()
