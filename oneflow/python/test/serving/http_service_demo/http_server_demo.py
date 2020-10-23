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
import oneflow.typing as tp

from imagenet1000_clsidx_to_labels import clsidx_2_labels
import resnet_model

from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
import json
import cv2
import numpy as np
import argparse


def get_parser():
    def float_list(x):
        return list(map(float, x.split(",")))

    parser = argparse.ArgumentParser("flags for http client demo")
    parser.add_argument("--server_address", type=str, default="localhost", help="")
    parser.add_argument("--server_port", type=int, default=8000, help="")
    parser.add_argument(
        "--model_load_dir", type=str, default=None, help="model load directory if need"
    )
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


def preprocess_image(im):
    im = cv2.resize(im.astype("uint8"), (224, 224))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype("float32")
    im = (im - args.rgb_mean) / args.rgb_std
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, "float32")


@flow.global_function("predict", flow.function_config())
def InferenceNet(
    images: tp.Numpy.Placeholder((1, 3, 224, 224), dtype=flow.float)
) -> tp.Numpy:
    logits = resnet_model.resnet50(images, training=False)
    predictions = flow.nn.softmax(logits)
    return predictions


check_point = flow.train.CheckPoint()
print("start load resnet50 model.")
check_point.load(args.model_load_dir)
print("load resnet50 model done.")


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Successfully connected!\n")

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)

        body = json.loads(body)
        list = json.loads(body["image"])
        image = np.array(list)
        print("\nserver received image shape: ", image.shape)

        predictions = InferenceNet(preprocess_image(np.array(list)))
        clsidx = predictions.argmax()
        print(
            "predicted class name: %s, prob: %f\n"
            % (clsidx_2_labels[clsidx], predictions.max())
        )

        self.send_response(200)
        self.end_headers()
        # response = BytesIO()
        # response.write(b'This is POST request. \n')
        # response.write(b'Received data len %d \n' % content_length)
        # self.wfile.write(response.getvalue())
        self.wfile.write(clsidx_2_labels[clsidx].encode())


httpd = HTTPServer((args.server_address, args.server_port), SimpleHTTPRequestHandler)
print("start server.")
httpd.serve_forever()
