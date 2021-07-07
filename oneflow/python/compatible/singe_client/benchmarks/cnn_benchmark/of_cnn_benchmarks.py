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
import benchmark_util
import data_loader
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
parser.add_argument("--learning_rate", type=float, default=1e-4, required=False)
parser.add_argument(
    "--optimizer", type=str, default="sgd", required=False, help="sgd, adam, momentum"
)
parser.add_argument(
    "--weight_l2",
    type=float,
    default=None,
    required=False,
    help="weight decay parameter",
)
parser.add_argument(
    "--iter_num", type=int, default=10, required=False, help="total iterations to run"
)
parser.add_argument(
    "--skip_iter_num",
    type=int,
    default=0,
    required=False,
    help="number of skipping iterations for benchmark purpose.",
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
    "--gpu_image_decoder",
    type=bool,
    default=False,
    required=False,
    help="Whether to use use ImageDecoderRandomCropResize.",
)
parser.add_argument(
    "--image_size", type=int, default=228, required=False, help="image size"
)

# log and resore/save
parser.add_argument(
    "--loss_print_every_n_iter",
    type=int,
    default=1,
    required=False,
    help="print loss every n iteration",
)
parser.add_argument(
    "--model_save_every_n_iter",
    type=int,
    default=200,
    required=False,
    help="save model every n iteration",
)
parser.add_argument(
    "--model_save_dir",
    type=str,
    default="./output/model_save-{}".format(
        str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    ),
    required=False,
    help="model save directory",
)
parser.add_argument(
    "--save_last_snapshot",
    type=bool,
    default=False,
    required=False,
    help="save model snapshot for last iteration",
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
parser.add_argument(
    "--enable_auto_mixed_precision",
    type=bool,
    default=False,
    required=False,
    help="automatically change the float net into mixed precision net",
)

args = parser.parse_args()


model_dict = {
    "resnet50": resnet_model.resnet50,
    "vgg16": vgg_model.vgg16,
    "alexnet": alexnet_model.alexnet,
}

optimizer_dict = {
    "sgd": {"naive_conf": {}},
    "adam": {"adam_conf": {"beta1": 0.9}},
    "momentum": {"momentum_conf": {"beta": 0.9}},
    "momentum-decay": {
        "momentum_conf": {"beta": 0.9},
        "learning_rate_decay": {
            "polynomial_conf": {"decay_batches": 300000, "end_learning_rate": 0.0001,},
        },
    },
}

#        "warmup_conf": {"linear_conf": {"warmup_batches":10000, "start_multiplier":0}},

func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.scope.consistent_view())
func_config.train.primary_lr(args.learning_rate)
func_config.default_data_type(flow.float)
func_config.train.model_update_conf(optimizer_dict[args.optimizer])
func_config.enable_auto_mixed_precision(args.enable_auto_mixed_precision)

if args.weight_l2:
    func_config.train.weight_l2(args.weight_l2)

flow.config.gpu_device_num(args.gpu_num_per_node)


@flow.global_function(func_config)
def TrainNet():

    total_device_num = args.node_num * args.gpu_num_per_node
    batch_size = total_device_num * args.batch_size_per_device

    if args.data_dir:
        assert os.path.exists(args.data_dir)
        print("Loading data from {}".format(args.data_dir))
        (labels, images) = data_loader.load_imagenet(
            args.data_dir,
            args.image_size,
            batch_size,
            args.data_part_num,
            args.gpu_image_decoder,
        )
    else:
        print("Loading synthetic data.")
        (labels, images) = data_loader.load_synthetic(args.image_size, batch_size)

    logits = model_dict[args.model](images)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, logits, name="softmax_loss"
    )
    flow.losses.add_loss(loss)

    return loss


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

    total_batch_size = (
        args.node_num * args.gpu_num_per_node * args.batch_size_per_device
    )
    speedometer = benchmark_util.CNNSpeedometer()
    start_time = time.time()

    for step in range(args.skip_iter_num + args.iter_num):
        cb = speedometer.speedometer_cb(
            step,
            start_time,
            total_batch_size,
            args.skip_iter_num,
            args.iter_num,
            args.loss_print_every_n_iter,
        )
        TrainNet().async_get(cb)

        if (step + 1) % args.model_save_every_n_iter == 0:
            if not os.path.exists(args.model_save_dir):
                os.makedirs(args.model_save_dir)
            snapshot_save_path = os.path.join(
                args.model_save_dir, "snapshot_%d" % (step + 1)
            )
            print("Saving model to {}.".format(snapshot_save_path))
            check_point.save(snapshot_save_path)

    if args.save_last_snapshot:
        snapshot_save_path = os.path.join(args.model_save_dir, "last_snapshot")
        if not os.path.exists(snapshot_save_path):
            os.makedirs(snapshot_save_path)
        print("Saving model to {}.".format(snapshot_save_path))
        check_point.save(snapshot_save_path)


if __name__ == "__main__":
    main()
