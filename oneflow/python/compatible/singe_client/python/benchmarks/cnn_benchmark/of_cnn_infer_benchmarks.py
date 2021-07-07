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
from __future__ import absolute_import, division, print_function

import argparse
import os
import time
from datetime import datetime

import alexnet_model
import data_loader
import inceptionv3_model
import oneflow as flow
import resnet_model
import vgg_model

parser = argparse.ArgumentParser(description="flags for cnn benchmark")

# resouce
parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("--node_num", type=int, default=1)
parser.add_argument(
    "--node_list",
    type=str,
    default=None,
    required=False,
    help="nodes' IP address, split by comma",
)

# train
parser.add_argument(
    "--model", type=str, default="vgg16", required=False, help="vgg16 or resnet50"
)
parser.add_argument("--batch_size_per_device", type=int, default=8, required=False)
parser.add_argument(
    "--iter_num", type=int, default=10, required=False, help="total iterations to run"
)
parser.add_argument(
    "--warmup_iter_num",
    type=int,
    default=0,
    required=False,
    help="total iterations to run",
)
parser.add_argument(
    "--data_dir", type=str, default=None, required=False, help="dataset directory"
)
parser.add_argument(
    "--data_part_num",
    type=int,
    default=32,
    required=False,
    help="data part number in dataset",
)
parser.add_argument(
    "--image_size", type=int, default=228, required=False, help="image size"
)

parser.add_argument(
    "--use_tensorrt",
    dest="use_tensorrt",
    action="store_true",
    default=False,
    required=False,
    help="inference with tensorrt",
)
parser.add_argument(
    "--use_xla_jit",
    dest="use_xla_jit",
    action="store_true",
    default=False,
    required=False,
    help="inference with xla jit",
)

parser.add_argument(
    "--precision",
    type=str,
    default="float32",
    required=False,
    help="inference with low precision",
)

# log and resore/save
parser.add_argument(
    "--print_every_n_iter",
    type=int,
    default=1,
    required=False,
    help="print log every n iterations",
)
parser.add_argument(
    "--model_load_dir",
    type=str,
    default=None,
    required=False,
    help="model load directory",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="./output",
    required=False,
    help="log info save directory",
)

args = parser.parse_args()

model_dict = {
    "resnet50": resnet_model.resnet50,
    "inceptionv3": inceptionv3_model.inceptionv3,
    "vgg16": vgg_model.vgg16,
    "alexnet": alexnet_model.alexnet,
}

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

flow.config.gpu_device_num(args.gpu_num_per_node)
if args.use_tensorrt:
    func_config.use_tensorrt()
if args.use_xla_jit:
    func_config.use_xla_jit()

if args.precision == "float16":
    if not args.use_tensorrt:
        func_config.enable_auto_mixed_precision()
    else:
        func_config.tensorrt.use_fp16()


@flow.global_function(func_config)
def InferenceNet():

    total_device_num = args.node_num * args.gpu_num_per_node
    batch_size = total_device_num * args.batch_size_per_device

    if args.data_dir:
        assert os.path.exists(args.data_dir)
        print("Loading data from {}".format(args.data_dir))
        (labels, images) = data_loader.load_imagenet(
            args.data_dir, args.image_size, batch_size, args.data_part_num
        )
    else:
        print("Loading synthetic data.")
        (labels, images) = data_loader.load_synthetic(args.image_size, batch_size)

    logits = model_dict[args.model](images)
    softmax = flow.nn.softmax(logits)
    return softmax


def main():
    print("=".ljust(66, "="))
    print(
        "Running {}: num_gpu_per_node = {}, num_nodes = {}.".format(
            args.model, args.gpu_num_per_node, args.node_num
        )
    )
    print("=".ljust(66, "="))
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("-".ljust(66, "-"))
    print("Time stamp: {}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))

    flow.env.log_dir(args.log_dir)

    if args.node_num > 1:
        nodes = []
        for n in args.node_list.strip().split(","):
            addr_dict = {}
            addr_dict["addr"] = n
            nodes.append(addr_dict)

        flow.env.machine(nodes)

    check_point = flow.train.CheckPoint()
    if args.model_load_dir:
        assert os.path.isdir(args.model_load_dir)
        print("Restoring model from {}.".format(args.model_load_dir))
        check_point.load(args.model_load_dir)
    else:
        print("Init model on demand.")
        check_point.init()

    # warmups
    print("Runing warm up for {} iterations.".format(args.warmup_iter_num))
    for step in range(args.warmup_iter_num):
        predictions = InferenceNet().get()

    main.total_time = 0.0
    main.batch_size = args.node_num * args.gpu_num_per_node * args.batch_size_per_device
    main.start_time = time.time()

    def create_callback(step):
        def callback(predictions):
            if step % args.print_every_n_iter == 0:
                cur_time = time.time()
                duration = cur_time - main.start_time
                main.total_time += duration
                main.start_time = cur_time
                images_per_sec = main.batch_size / duration
                print(
                    "iter {}, speed: {:.3f}(sec/batch), {:.3f}(images/sec)".format(
                        step, duration, images_per_sec
                    )
                )
                if step == args.iter_num - 1:
                    avg_img_per_sec = main.batch_size * args.iter_num / main.total_time
                    print("-".ljust(66, "-"))
                    print("average speed: {:.3f}(images/sec)".format(avg_img_per_sec))
                    print("-".ljust(66, "-"))

        return callback

    for step in range(args.iter_num):
        InferenceNet().async_get(create_callback(step))
        # predictions = InferenceNet().get()
        # create_callback(step)(predictions)
        # print(predictions)


if __name__ == "__main__":
    main()
