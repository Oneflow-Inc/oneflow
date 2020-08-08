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

_DATA_DIR = "/dataset/PNGS/PNG299/of_record_repeated"
_EVAL_DIR = _DATA_DIR
_TRAIN_DIR = _DATA_DIR
_MODEL_LOAD = "/dataset/PNGS/cnns_model_for_test/inceptionv3/models/of_model"
_MODEL_SAVE_DIR = "./model_save-{}".format(
    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
)
NODE_LIST = "192.168.1.12,192.168.1.14"


class DLNetSpec(object):
    def __init__(self, enable_auto_mixed_precision):
        self.batch_size = 8
        self.data_part_num = 32
        self.eval_dir = _DATA_DIR
        self.train_dir = _DATA_DIR
        self.model_save_dir = _MODEL_SAVE_DIR
        self.model_load_dir = _MODEL_LOAD
        self.num_nodes = 1
        self.gpu_num_per_node = 1
        self.iter_num = 10
        self.enable_auto_mixed_precision = enable_auto_mixed_precision


parser = argparse.ArgumentParser(description="flags for multi-node and resource")
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-i", "--iter_num", type=int, default=10, required=False)
parser.add_argument("-b", "--batch_size", type=int, default=8, required=False)
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

# TODO: add this interface to oneflow.layers
def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kSigmoid,
    use_bias=True,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=flow.constant_initializer(),
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    else:
        kernel_size = tuple(kernel_size)
    weight_shape = (filters, input.shape[1]) + kernel_size
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
    )
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
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.math.relu(output)
        elif activation == op_conf_util.kSigmoid:
            output = flow.math.sigmoid(output)
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
    rsz = flow.image.resize(image, resize_x=299, resize_y=299, color_space="RGB")
    normal = flow.image.crop_mirror_normalize(
        rsz,
        color_space="RGB",
        output_layout="NCHW",
        mean=rgb_mean,
        output_dtype=flow.float,
    )
    return normal, label


def InceptionA(in_blob, index):
    with flow.scope.namespace("mixed_{}".format(index)):
        with flow.scope.namespace("branch1x1"):
            branch1x1 = _conv2d_layer(
                "conv0", in_blob, filters=64, kernel_size=1, strides=1, padding="SAME"
            )
        with flow.scope.namespace("branch5x5"):
            branch5x5_1 = _conv2d_layer(
                "conv0", in_blob, filters=48, kernel_size=1, strides=1, padding="SAME"
            )
            branch5x5_2 = _conv2d_layer(
                "conv1",
                branch5x5_1,
                filters=64,
                kernel_size=5,
                strides=1,
                padding="SAME",
            )
        with flow.scope.namespace("branch3x3dbl"):
            branch3x3dbl_1 = _conv2d_layer(
                "conv0", in_blob, filters=64, kernel_size=1, strides=1, padding="SAME"
            )
            branch3x3dbl_2 = _conv2d_layer(
                "conv1",
                branch3x3dbl_1,
                filters=96,
                kernel_size=3,
                strides=1,
                padding="SAME",
            )
            branch3x3dbl_3 = _conv2d_layer(
                "conv2",
                branch3x3dbl_2,
                filters=96,
                kernel_size=3,
                strides=1,
                padding="SAME",
            )
        with flow.scope.namespace("branch_pool"):
            branch_pool_1 = flow.nn.avg_pool2d(
                in_blob,
                ksize=3,
                strides=1,
                padding="SAME",
                data_format="NCHW",
                name="pool",
            )
            branch_pool_2 = _conv2d_layer(
                "conv",
                branch_pool_1,
                filters=32 if index == 0 else 64,
                kernel_size=1,
                strides=1,
                padding="SAME",
            )

        inceptionA_bn = []
        inceptionA_bn.append(branch1x1)
        inceptionA_bn.append(branch5x5_2)
        inceptionA_bn.append(branch3x3dbl_3)
        inceptionA_bn.append(branch_pool_2)

        mixed_concat = flow.concat(values=inceptionA_bn, axis=1, name="concat")

    return mixed_concat


def InceptionB(in_blob, index):
    with flow.scope.namespace("mixed_{}".format(index)):
        with flow.scope.namespace("branch3x3"):
            branch3x3 = _conv2d_layer(
                "conv0", in_blob, filters=384, kernel_size=3, strides=2, padding="VALID"
            )
        with flow.scope.namespace("branch3x3dbl"):
            branch3x3dbl_1 = _conv2d_layer(
                "conv0", in_blob, filters=64, kernel_size=1, strides=1, padding="SAME"
            )
            branch3x3dbl_2 = _conv2d_layer(
                "conv1",
                branch3x3dbl_1,
                filters=96,
                kernel_size=3,
                strides=1,
                padding="SAME",
            )
            branch3x3dbl_3 = _conv2d_layer(
                "conv2",
                branch3x3dbl_2,
                filters=96,
                kernel_size=3,
                strides=2,
                padding="VALID",
            )
        with flow.scope.namespace("branch_pool"):
            branch_pool = flow.nn.max_pool2d(
                in_blob,
                ksize=3,
                strides=2,
                padding="VALID",
                data_format="NCHW",
                name="pool0",
            )

        inceptionB_bn = []
        inceptionB_bn.append(branch3x3)
        inceptionB_bn.append(branch3x3dbl_3)
        inceptionB_bn.append(branch_pool)
        mixed_concat = flow.concat(values=inceptionB_bn, axis=1, name="concat")

    return mixed_concat


def InceptionC(in_blob, index, filters):
    with flow.scope.namespace("mixed_{}".format(index)):
        with flow.scope.namespace("branch1x1"):
            branch1x1 = _conv2d_layer(
                "conv0", in_blob, filters=192, kernel_size=1, strides=1, padding="SAME"
            )
        with flow.scope.namespace("branch7x7"):
            branch7x7_1 = _conv2d_layer(
                "conv0",
                in_blob,
                filters=filters,
                kernel_size=1,
                strides=1,
                padding="SAME",
            )
            branch7x7_2 = _conv2d_layer(
                "conv1",
                branch7x7_1,
                filters=filters,
                kernel_size=[1, 7],
                strides=1,
                padding="SAME",
            )
            branch7x7_3 = _conv2d_layer(
                "conv2",
                branch7x7_2,
                filters=192,
                kernel_size=[7, 1],
                strides=[1, 1],
                padding="SAME",
            )
        with flow.scope.namespace("branch7x7dbl"):
            branch7x7dbl_1 = _conv2d_layer(
                "conv0",
                in_blob,
                filters=filters,
                kernel_size=1,
                strides=1,
                padding="SAME",
            )
            branch7x7dbl_2 = _conv2d_layer(
                "conv1",
                branch7x7dbl_1,
                filters=filters,
                kernel_size=[7, 1],
                strides=1,
                padding="SAME",
            )
            branch7x7dbl_3 = _conv2d_layer(
                "conv2",
                branch7x7dbl_2,
                filters=filters,
                kernel_size=[1, 7],
                strides=1,
                padding="SAME",
            )
            branch7x7dbl_4 = _conv2d_layer(
                "conv3",
                branch7x7dbl_3,
                filters=filters,
                kernel_size=[7, 1],
                strides=1,
                padding="SAME",
            )
            branch7x7dbl_5 = _conv2d_layer(
                "conv4",
                branch7x7dbl_4,
                filters=192,
                kernel_size=[1, 7],
                strides=1,
                padding="SAME",
            )
        with flow.scope.namespace("branch_pool"):
            branch_pool_1 = flow.nn.avg_pool2d(
                in_blob,
                ksize=3,
                strides=1,
                padding="SAME",
                data_format="NCHW",
                name="pool",
            )
            branch_pool_2 = _conv2d_layer(
                "conv",
                branch_pool_1,
                filters=192,
                kernel_size=[1, 1],
                strides=1,
                padding="SAME",
            )

        inceptionC_bn = []
        inceptionC_bn.append(branch1x1)
        inceptionC_bn.append(branch7x7_3)
        inceptionC_bn.append(branch7x7dbl_5)
        inceptionC_bn.append(branch_pool_2)
        mixed_concat = flow.concat(values=inceptionC_bn, axis=1, name="concat")

    return mixed_concat


def InceptionD(in_blob, index):
    with flow.scope.namespace("mixed_{}".format(index)):
        with flow.scope.namespace("branch3x3"):
            branch3x3_1 = _conv2d_layer(
                "conv0", in_blob, filters=192, kernel_size=1, strides=1, padding="SAME"
            )
            branch3x3_2 = _conv2d_layer(
                "conv1",
                branch3x3_1,
                filters=320,
                kernel_size=3,
                strides=2,
                padding="VALID",
            )
        with flow.scope.namespace("branch7x7x3"):
            branch7x7x3_1 = _conv2d_layer(
                "conv0", in_blob, filters=192, kernel_size=1, strides=1, padding="SAME"
            )
            branch7x7x3_2 = _conv2d_layer(
                "conv1",
                branch7x7x3_1,
                filters=192,
                kernel_size=[1, 7],
                strides=1,
                padding="SAME",
            )
            branch7x7x3_3 = _conv2d_layer(
                "conv2",
                branch7x7x3_2,
                filters=192,
                kernel_size=[7, 1],
                strides=1,
                padding="SAME",
            )
            branch7x7x3_4 = _conv2d_layer(
                "conv3",
                branch7x7x3_3,
                filters=192,
                kernel_size=3,
                strides=2,
                padding="VALID",
            )
        with flow.scope.namespace("branch_pool"):
            branch_pool = flow.nn.max_pool2d(
                in_blob,
                ksize=3,
                strides=2,
                padding="VALID",
                data_format="NCHW",
                name="pool",
            )

        inceptionD_bn = []
        inceptionD_bn.append(branch3x3_2)
        inceptionD_bn.append(branch7x7x3_4)
        inceptionD_bn.append(branch_pool)

        mixed_concat = flow.concat(values=inceptionD_bn, axis=1, name="concat")

    return mixed_concat


def InceptionE(in_blob, index):
    with flow.scope.namespace("mixed_{}".format(index)):
        with flow.scope.namespace("branch1x1"):
            branch1x1 = _conv2d_layer(
                "conv0", in_blob, filters=320, kernel_size=1, strides=1, padding="SAME"
            )
        with flow.scope.namespace("branch3x3"):
            branch3x3_1 = _conv2d_layer(
                "conv0", in_blob, filters=384, kernel_size=1, strides=1, padding="SAME"
            )
            branch3x3_2 = _conv2d_layer(
                "conv1",
                branch3x3_1,
                filters=384,
                kernel_size=[1, 3],
                strides=1,
                padding="SAME",
            )
            branch3x3_3 = _conv2d_layer(
                "conv2",
                branch3x3_1,
                filters=384,
                kernel_size=[3, 1],
                strides=[1, 1],
                padding="SAME",
            )
            inceptionE_1_bn = []
            inceptionE_1_bn.append(branch3x3_2)
            inceptionE_1_bn.append(branch3x3_3)
            concat_branch3x3 = flow.concat(
                values=inceptionE_1_bn, axis=1, name="concat"
            )
        with flow.scope.namespace("branch3x3dbl"):
            branch3x3dbl_1 = _conv2d_layer(
                "conv0", in_blob, filters=448, kernel_size=1, strides=1, padding="SAME"
            )
            branch3x3dbl_2 = _conv2d_layer(
                "conv1",
                branch3x3dbl_1,
                filters=384,
                kernel_size=3,
                strides=1,
                padding="SAME",
            )
            branch3x3dbl_3 = _conv2d_layer(
                "conv2",
                branch3x3dbl_2,
                filters=384,
                kernel_size=[1, 3],
                strides=1,
                padding="SAME",
            )
            branch3x3dbl_4 = _conv2d_layer(
                "conv3",
                branch3x3dbl_2,
                filters=384,
                kernel_size=[3, 1],
                strides=1,
                padding="SAME",
            )
            inceptionE_2_bn = []
            inceptionE_2_bn.append(branch3x3dbl_3)
            inceptionE_2_bn.append(branch3x3dbl_4)
            concat_branch3x3dbl = flow.concat(
                values=inceptionE_2_bn, axis=1, name="concat"
            )
        with flow.scope.namespace("branch_pool"):
            branch_pool_1 = flow.nn.avg_pool2d(
                in_blob,
                ksize=3,
                strides=1,
                padding="SAME",
                data_format="NCHW",
                name="pool",
            )
            branch_pool_2 = _conv2d_layer(
                "conv",
                branch_pool_1,
                filters=192,
                kernel_size=[1, 1],
                strides=1,
                padding="SAME",
            )

        inceptionE_total_bn = []
        inceptionE_total_bn.append(branch1x1)
        inceptionE_total_bn.append(concat_branch3x3)
        inceptionE_total_bn.append(concat_branch3x3dbl)
        inceptionE_total_bn.append(branch_pool_2)

        concat_total = flow.concat(values=inceptionE_total_bn, axis=1, name="concat")

    return concat_total


def InceptionV3(images, labels, trainable=True):
    conv0 = _conv2d_layer(
        "conv0", images, filters=32, kernel_size=3, strides=2, padding="VALID"
    )
    conv1 = _conv2d_layer(
        "conv1", conv0, filters=32, kernel_size=3, strides=1, padding="VALID"
    )
    conv2 = _conv2d_layer(
        "conv2", conv1, filters=64, kernel_size=3, strides=1, padding="SAME"
    )
    pool1 = flow.nn.max_pool2d(
        conv2, ksize=3, strides=2, padding="VALID", data_format="NCHW", name="pool1"
    )
    conv3 = _conv2d_layer(
        "conv3", pool1, filters=80, kernel_size=1, strides=1, padding="VALID"
    )
    conv4 = _conv2d_layer(
        "conv4", conv3, filters=192, kernel_size=3, strides=1, padding="VALID"
    )
    pool2 = flow.nn.max_pool2d(
        conv4, ksize=3, strides=2, padding="VALID", data_format="NCHW", name="pool2"
    )

    # mixed_0 ~ mixed_2
    mixed_0 = InceptionA(pool2, 0)
    mixed_1 = InceptionA(mixed_0, 1)
    mixed_2 = InceptionA(mixed_1, 2)

    # mixed_3
    mixed_3 = InceptionB(mixed_2, 3)

    # mixed_4 ~ mixed_7
    mixed_4 = InceptionC(mixed_3, 4, 128)
    mixed_5 = InceptionC(mixed_4, 5, 160)
    mixed_6 = InceptionC(mixed_5, 6, 160)
    mixed_7 = InceptionC(mixed_6, 7, 192)

    # mixed_8
    mixed_8 = InceptionD(mixed_7, 8)

    # mixed_9 ~ mixed_10
    mixed_9 = InceptionE(mixed_8, 9)
    mixed_10 = InceptionE(mixed_9, 10)

    # pool3
    pool3 = flow.nn.avg_pool2d(
        mixed_10, ksize=8, strides=1, padding="VALID", data_format="NCHW", name="pool3"
    )

    with flow.scope.namespace("logits"):
        pool3 = flow.reshape(pool3, [pool3.shape[0], -1])
        # TODO: Need to transpose weight when converting model from TF to OF if
        # you want to use layers.dense interface.
        # fc1 = flow.layers.dense(
        #     pool3,
        #     1001,
        #     activation=None,
        #     use_bias=False,
        #     kernel_initializer=flow.truncated_normal(0.816496580927726),
        #     bias_initializer=flow.constant_initializer(),
        #     name="fc1",
        # )
        weight = flow.get_variable(
            "fc1-weight",
            shape=(pool3.shape[1], 1001),
            dtype=flow.float,
            initializer=flow.truncated_normal(0.816496580927726),
            model_name="weight",
        )
        bias = flow.get_variable(
            "fc1-bias",
            shape=(1001,),
            dtype=flow.float,
            initializer=flow.constant_initializer(),
            model_name="bias",
        )
        fc1 = flow.matmul(pool3, weight)
        fc1 = flow.nn.bias_add(fc1, bias)

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=fc1, name="softmax_loss"
    )

    return loss


def main(args):
    flow.config.machine_num(args.num_nodes)
    flow.config.gpu_device_num(args.gpu_num_per_node)
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.default_data_type(flow.float)
    func_config.enable_auto_mixed_precision(args.enable_auto_mixed_precision)

    @flow.global_function(type="train", function_config=func_config)
    def TrainNet():
        (images, labels) = _data_load_layer(args, args.train_dir)
        loss = InceptionV3(images, labels)
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [0.0001]), momentum=0
        ).minimize(loss)
        return loss

    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)

    num_nodes = args.num_nodes
    print(
        "Traning inceptionv3: num_gpu_per_node = {}, num_nodes = {}.".format(
            args.gpu_num_per_node, num_nodes
        )
    )

    print("{:>12}  {:>12}  {:>12}".format("iter", "loss type", "loss value"))
    loss = []
    for i in range(args.iter_num):
        train_loss = TrainNet().get().mean()
        loss.append(train_loss)

        fmt_str = "{:>12}  {:>12}  {:>12.6f}"
        print(fmt_str.format(i, "train loss:", train_loss))

        if (i + 1) % 100 == 0:
            check_point.save(_MODEL_SAVE_DIR + str(i))

    # save loss to file
    loss_file = "{}n{}c.npy".format(
        str(num_nodes), str(args.gpu_num_per_node * num_nodes)
    )
    loss_path = "./of_loss/inceptionv3"
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    numpy.save(os.path.join(loss_path, loss_file), loss)


if __name__ == "__main__":
    args = parser.parse_args()
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
