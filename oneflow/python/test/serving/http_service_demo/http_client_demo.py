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
import argparse
import cv2
import http.client
import json
import os
import time


def get_parser():
    parser = argparse.ArgumentParser("flags for http client demo")
    parser.add_argument("--server_address", type=str, default="localhost", help="")
    parser.add_argument("--server_port", type=int, default=8000, help="")
    parser.add_argument(
        "--test_images_path",
        type=str,
        default="/dataset/http_service_demo_client_images/",
        help="",
    )
    return parser


parser = get_parser()
args = parser.parse_args()

if not os.path.exists(args.test_images_path):
    raise Exception("Images path does not existed.")

conn = http.client.HTTPConnection(args.server_address, args.server_port)
conn.request("GET", "/")
r1 = conn.getresponse()
print(r1.status, r1.reason)
print(r1.read().decode())

headers = {"Content-type": "application/json"}
images = os.listdir(args.test_images_path)

while True:
    for im in images:
        img = cv2.imread(os.path.join(args.test_images_path, im))
        res = {"image": json.dumps(img.tolist())}
        json_data = json.dumps(res)

        print("\n##############################")
        print("send image %s to server" % im)
        conn.request("POST", "/post", json_data, headers)
        response = conn.getresponse()
        print("get predicted result from server: %s" % response.read().decode())
        print("##############################\n")

        time.sleep(2)

# conn.close()
