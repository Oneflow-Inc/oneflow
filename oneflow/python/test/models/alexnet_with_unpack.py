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
import os
from datetime import datetime

import numpy
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util

# _DATA_DIR = "/dataset/imagenet_227/train/32"
_DATA_DIR = "/dataset/PNGS/PNG227/of_record_repeated"
_MODEL_SAVE_DIR = "./model_save-{}".format(
    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
)
_MODEL_LOAD = "/dataset/PNGS/cnns_model_for_test/alexnet/models/of_model_bk"
NODE_LIST = "192.168.1.12,192.168.1.14"


class DLNetSpec(object):
    def __init__(self):
        self.batch_size = 8
        self.data_part_num = 32
        self.eval_dir = _DATA_DIR
        self.train_dir = _DATA_DIR
        self.model_save_dir = _MODEL_SAVE_DIR
        self.model_load_dir = _MODEL_LOAD
        self.num_nodes = 1
        self.gpu_num_per_node = 1
        self.iter_num = 10
        self.num_unpack = 2


parser = argparse.ArgumentParser(description="flags for multi-node and resource")
parser.add_argument("-nn", "--num_nodes", type=str, default=1, required=False)
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-i", "--iter_num", type=int, default=10, required=False)
parser.add_argument(
    "-m", "--multinode", default=False, action="store_true", required=False
)
parser.add_argument("-n", "--node_list", type=str, default=NODE_LIST, required=False)
parser.add_argument(
    "-s", "--skip_scp_binary", default=False, action="store_true", required=False
)
parser.add_argument(
    "-c",
    "--scp_binary_without_uuid",
    default=False,
    action="store_true",
    required=False,
)
parser.add_argument(
    "-r", "--remote_by_hand", default=False, action="store_true", required=False
)
parser.add_argument("-e", "--eval_dir", type=str, default=_DATA_DIR, required=False)
parser.add_argument("-t", "--train_dir", type=str, default=_DATA_DIR, required=False)
parser.add_argument(
    "-load", "--model_load_dir", type=str, default=_MODEL_LOAD, required=False
)
parser.add_argument(
    "-save", "--model_save_dir", type=str, default=_MODEL_SAVE_DIR, required=False
)
parser.add_argument("-dn", "--data_part_num", type=int, default=32, required=False)
parser.add_argument("-b", "--batch_size", type=int, default=8, required=False)
parser.add_argument("-p", "--num_piece_in_batch", type=int, default=2, required=False)


def _conv2d_layer(
    args,
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=False,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=flow.random_uniform_initializer(),
):
    weight_shape = (filters, input.shape[1], kernel_size, kernel_size)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
    )
    weight = flow.identity(weight)
    weight = flow.repeat(weight, args.num_piece_in_batch)
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
        )
        bias = flow.identity(bias)
        bias = flow.repeat(bias, args.num_piece_in_batch)
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.math.relu(output)
        else:
            raise NotImplementedError

    return output


def _data_load_layer(args, data_dir):
    node_num = args.num_nodes
    total_batch_size = args.batch_size * args.gpu_num_per_node * node_num
    rgb_mean = [123.68, 116.78, 103.94]
    ofrecord = flow.data.ofrecord_reader(
        data_dir,
        batch_size=total_batch_size,
        data_part_num=args.data_part_num,
        name="decode",
    )
    image = flow.data.ofrecord_image_decoder(ofrecord, "encoded", color_space="RGB")
    label = flow.data.ofrecord_raw_decoder(
        ofrecord, "class/label", shape=(), dtype=flow.int32
    )
    rsz = flow.image.resize(image, resize_x=227, resize_y=227, color_space="RGB")
    normal = flow.image.crop_mirror_normalize(
        rsz,
        color_space="RGB",
        output_layout="NCHW",
        mean=rgb_mean,
        output_dtype=flow.float,
    )
    return (
        flow.unpack(label, args.num_piece_in_batch),
        flow.unpack(normal, args.num_piece_in_batch),
    )


def _dense_layer(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=None,
    trainable=True,
    name=None,
):
    in_shape = inputs.shape
    in_num_axes = len(in_shape)
    assert in_num_axes >= 2

    name_prefix = name if name is not None else id_util.UniqueStr("Dense_")
    inputs = flow.reshape(inputs, (-1, in_shape[-1])) if in_num_axes > 2 else inputs

    weight = flow.get_variable(
        name="{}-weight".format(name_prefix),
        shape=(units, inputs.shape[1]),
        dtype=inputs.dtype,
        initializer=(
            kernel_initializer
            if kernel_initializer is not None
            else flow.constant_initializer(0)
        ),
        trainable=trainable,
        model_name="weight",
    )
    weight = flow.identity(weight)
    weight = flow.repeat(weight, args.num_piece_in_batch)

    out = flow.matmul(
        a=inputs, b=weight, transpose_b=True, name="{}_matmul".format(name_prefix),
    )
    if use_bias:
        bias = flow.get_variable(
            name="{}-bias".format(name_prefix),
            shape=(units,),
            dtype=inputs.dtype,
            initializer=(
                bias_initializer
                if bias_initializer is not None
                else flow.constant_initializer(0)
            ),
            trainable=trainable,
            model_name="bias",
        )

        bias = flow.identity(bias)
        bias = flow.repeat(bias, args.num_piece_in_batch)

        out = flow.nn.bias_add(out, bias, name="{}_bias_add".format(name_prefix))
    out = (
        activation(out, name="{}_activation".format(name_prefix))
        if activation is not None
        else out
    )
    out = flow.reshape(out, in_shape[:-1] + (units,)) if in_num_axes > 2 else out

    return out


def alexnet(args, images, labels, trainable=True):
    conv1 = _conv2d_layer(
        args, "conv1", images, filters=64, kernel_size=11, strides=4, padding="VALID",
    )

    pool1 = flow.nn.avg_pool2d(conv1, 3, 2, "VALID", "NCHW", name="pool1")

    conv2 = _conv2d_layer(args, "conv2", pool1, filters=192, kernel_size=5)

    pool2 = flow.nn.avg_pool2d(conv2, 3, 2, "VALID", "NCHW", name="pool2")

    conv3 = _conv2d_layer(args, "conv3", pool2, filters=384)

    conv4 = _conv2d_layer(args, "conv4", conv3, filters=384)

    conv5 = _conv2d_layer(args, "conv5", conv4, filters=256)

    pool5 = flow.nn.avg_pool2d(conv5, 3, 2, "VALID", "NCHW", name="pool5")

    def _get_initializer():
        kernel_initializer = op_conf_util.InitializerConf()
        kernel_initializer.truncated_normal_conf.std = 0.816496580927726
        return kernel_initializer

    if len(pool5.shape) > 2:
        pool5 = flow.reshape(pool5, shape=(pool5.shape[0], -1))

    fc1 = _dense_layer(
        inputs=pool5,
        units=4096,
        activation=flow.math.relu,
        use_bias=False,
        kernel_initializer=_get_initializer(),
        bias_initializer=False,
        trainable=trainable,
        name="fc1",
    )

    dropout1 = fc1

    fc2 = _dense_layer(
        inputs=dropout1,
        units=4096,
        activation=flow.math.relu,
        use_bias=False,
        kernel_initializer=_get_initializer(),
        bias_initializer=False,
        trainable=trainable,
        name="fc2",
    )

    dropout2 = fc2

    fc3 = _dense_layer(
        inputs=dropout2,
        units=1001,
        activation=None,
        use_bias=False,
        kernel_initializer=_get_initializer(),
        bias_initializer=False,
        trainable=trainable,
        name="fc3",
    )

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, fc3, name="softmax_loss"
    )

    return loss


def main(args):
    flow.config.machine_num(args.num_nodes)
    flow.config.gpu_device_num(args.gpu_num_per_node)

    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.default_data_type(flow.float)
    func_config.cudnn_conv_force_fwd_algo(0)
    func_config.cudnn_conv_force_bwd_data_algo(1)
    func_config.cudnn_conv_force_bwd_filter_algo(1)

    @flow.global_function(type="train", function_config=func_config)
    def alexnet_train_job():
        (labels, images) = _data_load_layer(args, args.train_dir)
        loss = alexnet(args, images, labels)
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [0.00001]), momentum=0
        ).minimize(loss)
        return flow.pack(loss, args.num_piece_in_batch)

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def alexnet_eval_job():
        with flow.scope.consistent_view():
            (labels, images) = _data_load_layer(args, args.eval_dir)
            loss = alexnet(args, images, labels)
            return flow.pack(loss, args.num_piece_in_batch)

    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)

    num_nodes = args.num_nodes
    print(
        "Traning alexnet: num_gpu_per_node = {}, num_nodes = {}.".format(
            args.gpu_num_per_node, num_nodes
        )
    )

    print("{:>12}  {:>12}  {:>12}".format("iter", "loss type", "loss value"))
    loss = []
    for i in range(args.iter_num):
        train_loss = alexnet_train_job().get().mean()
        loss.append(train_loss)

        fmt_str = "{:>12}  {:>12}  {:>12.6f}"
        print(fmt_str.format(i, "train loss:", train_loss))

        # if (i + 1) % 10 == 0:
        #   eval_loss = alexnet_eval_job().get().mean()
        # print(
        #     fmt_str.format(
        #         i, "eval loss:", eval_loss
        #     )
        # )
        if (i + 1) % 100 == 0:
            check_point.save(_MODEL_SAVE_DIR + str(i))

    # save loss to file
    loss_file = "{}n{}c.npy".format(
        str(num_nodes), str(args.gpu_num_per_node * num_nodes)
    )
    loss_path = "./of_loss/alexnet"
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    numpy.save(os.path.join(loss_path, loss_file), loss)


if __name__ == "__main__":
    args = parser.parse_args()
    args.num_nodes = len(args.node_list.strip().split(",")) if args.multinode else 1
    flow.env.ctrl_port(9788)
    if args.multinode:
        flow.env.ctrl_port(12138)
        nodes = []
        for n in args.node_list.strip().split(","):
            addr_dict = {}
            addr_dict["addr"] = n
            nodes.append(addr_dict)

        flow.env.machine(nodes)
        if args.remote_by_hand is False:
            if args.scp_binary_without_uuid:
                flow.deprecated.init_worker(scp_binary=True, use_uuid=False)
            elif args.skip_scp_binary:
                flow.deprecated.init_worker(scp_binary=False, use_uuid=False)
            else:
                flow.deprecated.init_worker(scp_binary=True, use_uuid=True)

    main(args)
    if (
        args.multinode
        and args.skip_scp_binary is False
        and args.scp_binary_without_uuid is False
    ):
        flow.deprecated.delete_worker()
