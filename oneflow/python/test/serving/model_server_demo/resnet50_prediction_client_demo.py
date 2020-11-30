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
import time
import oneflow as flow


def get_parser():
    parser = argparse.ArgumentParser("flags for grpc client demo")
    parser.add_argument("--server_address", type=str, default="localhost", help="")
    parser.add_argument("--server_port", type=int, default=9887, help="")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/dataset/ImageNet/ofrecord/validation",
        help="",
    )
    parser.add_argument("--data_part_num", type=int, default=256, help="")
    parser.add_argument("--batch_size", type=int, default=4, help="")
    parser.add_argument("--image_size", type=int, default=224, help="")
    parser.add_argument("--data_format", type=str, default="NCHW", help="")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    client = flow.serving.PredictionGrpcClient(args.server_address, args.server_port)
    dataset = flow.serving.ImageNetRecordDataset(
        data_dir=args.data_dir,
        num_data_parts=args.data_part_num,
        batch_size=args.batch_size,
        shuffle_data_part=False,
        image_resize_size=args.image_size,
        data_format=args.data_format,
    )

    num_batchs = 10
    model_name = "resnet50"
    graph_name = None

    for i, (image, label) in enumerate(dataset):
        print("====> iter{} <====".format(i))
        print("send image to server")
        response = client.predict("resnet50", graph_name, image=image, label=label)
        print("get result from server:")
        for output_name, tensor_proto in response.outputs.items():
            arr = flow.serving.tensor_proto_to_array(tensor_proto)
            print(
                "{} shape: {}, dtype: {}, data:\n{}".format(
                    output_name, arr.shape, arr.dtype, arr
                )
            )

        i += 1
        if i > 10:
            break
        else:
            time.sleep(2)
