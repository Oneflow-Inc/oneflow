import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from datetime import datetime
import argparse
import os
import numpy

_DATA_DIR = "/dataset/PNGS/PNG224/of_record_repeated"
_SINGLE_DATA_DIR = "/dataset/PNGS/PNG224/of_record"
_MODEL_LOAD_DIR = "/dataset/PNGS/cnns_model_for_test/vgg16/models/of_model"
_MODEL_SAVE_DIR = "./model_save-{}".format(
    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
)
NODE_LIST = "192.168.1.12,192.168.1.14"

class DLNetSpec(object):
  def __init__(self):
    self.batch_size = 8
    self.data_part_num = 32
    self.eval_dir = _DATA_DIR
    self.train_dir = _DATA_DIR
    self.model_save_dir = _MODEL_SAVE_DIR
    self.model_load_dir = _MODEL_LOAD_DIR
    self.num_nodes = 1
    self.gpu_num_per_node = 1
    self.iter_num = 10

parser = argparse.ArgumentParser(
    description="flags for multi-node and resource")
parser.add_argument("-g", "--gpu_num_per_node",
                    type=int, default=1, required=False)
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
parser.add_argument("-e", "--eval_dir", type=str,
                    default=_DATA_DIR, required=False)
parser.add_argument("-t", "--train_dir", type=str,
                    default=_DATA_DIR, required=False)
parser.add_argument("-load", "--model_load_dir", type=str,
                    default=_MODEL_LOAD_DIR, required=False)
parser.add_argument(
    "-save", "--model_save_dir", type=str, default=_MODEL_SAVE_DIR, required=False
)
parser.add_argument("-dn", "--data_part_num", type=int, default=32, required=False)
parser.add_argument("-b", "--batch_size", type=int, default=8, required=False)


def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=True,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=flow.constant_initializer(),
):
    weight_shape = (filters, input.static_shape[1], kernel_size, kernel_size)
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
        output = flow.nn.bias_add(output, bias, "NCHW")
    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.keras.activations.relu(output)
        else:
            raise NotImplementedError

    return output


def _data_load_layer(args, data_dir):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(224, 224, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec([flow.data.ImagePreprocessor("bgr2rgb")]),
        preprocessors=[flow.data.NormByChannelPreprocessor(
            (123.68, 116.78, 103.94))],
    )

    label_blob_conf = flow.data.BlobConf(
        "class/label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    node_num = args.num_nodes
    total_batch_size = args.batch_size * args.gpu_num_per_node * node_num
    return flow.data.decode_ofrecord(
        data_dir, (label_blob_conf, image_blob_conf),
        batch_size=total_batch_size, data_part_num=args.data_part_num, name="decode",
    )


def _conv_block(in_blob, index, filters, conv_times):
    conv_block = []
    conv_block.insert(0, in_blob)
    for i in range(conv_times):
        conv_i = _conv2d_layer(
            name="conv{}".format(index),
            input=conv_block[i],
            filters=filters,
            kernel_size=3,
            strides=1,
        )
        conv_block.append(conv_i)
        index += 1

    return conv_block


def vgg(images, labels, trainable=True):
    to_return = []
    transposed = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])
    conv1 = _conv_block(transposed, 0, 64, 2)
    pool1 = flow.nn.max_pool2d(conv1[-1], 2, 2, "VALID", "NCHW", name="pool1")

    conv2 = _conv_block(pool1, 2, 128, 2)
    pool2 = flow.nn.max_pool2d(conv2[-1], 2, 2, "VALID", "NCHW", name="pool2")

    conv3 = _conv_block(pool2, 4, 256, 3)
    pool3 = flow.nn.max_pool2d(conv3[-1], 2, 2, "VALID", "NCHW", name="pool3")

    conv4 = _conv_block(pool3, 7, 512, 3)
    pool4 = flow.nn.max_pool2d(conv4[-1], 2, 2, "VALID", "NCHW", name="pool4")

    conv5 = _conv_block(pool4, 10, 512, 3)
    pool5 = flow.nn.max_pool2d(conv5[-1], 2, 2, "VALID", "NCHW", name="pool5")

    def _get_kernel_initializer():
        kernel_initializer = op_conf_util.InitializerConf()
        kernel_initializer.truncated_normal_conf.std = 0.816496580927726
        return kernel_initializer

    def _get_bias_initializer():
        bias_initializer = op_conf_util.InitializerConf()
        bias_initializer.constant_conf.value = 0.0
        return bias_initializer

    pool5 = flow.reshape(pool5, [-1, 512])

    fc6 = flow.layers.dense(
        inputs=pool5,
        units=4096,
        activation=flow.keras.activations.relu,
        use_bias=True,
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        trainable=trainable,
        name="fc1"
    )

    fc7 = flow.layers.dense(
        inputs=fc6,
        units=4096,
        activation=flow.keras.activations.relu,
        use_bias=True,
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        trainable=trainable,
        name="fc2"
    )

    fc8 = flow.layers.dense(
        inputs=fc7,
        units=1001,
        use_bias=True,
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        trainable=trainable,
        name="fc_final"
    )

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, fc8, name="softmax_loss"
    )

    to_return.append(loss)
    return tuple(to_return)


def main(args):
    @flow.function
    def vgg_train_job():
        flow.config.train.primary_lr(0.00001)
        flow.config.train.model_update_conf(dict(naive_conf={}))
        (labels, images) = _data_load_layer(args, args.train_dir)
        to_return = vgg(images, labels)
        loss = to_return[-1]
        flow.losses.add_loss(loss)
        return loss

    @flow.function
    def vgg_eval_job():
        (labels, images) = _data_load_layer(args, args.eval_dir)
        return vgg(images, labels, False)
    flow.config.machine_num(args.num_nodes)
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.config.default_data_type(flow.float)
    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)

    num_nodes = args.num_nodes
    print("Traning vgg16: num_gpu_per_node = {}, num_nodes = {}.".format(args.gpu_num_per_node, num_nodes))

    print("{:>12}  {:>12}  {:>12}".format("iter", "loss type", "loss value"))
    loss = []
    for i in range(args.iter_num):
      train_loss = vgg_train_job().get().mean()
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
    loss_file = "{}n{}c.npy".format(str(num_nodes), str(args.gpu_num_per_node * num_nodes))
    loss_path = "./of_loss/vgg16"
    if not os.path.exists(loss_path): os.makedirs(loss_path)
    numpy.save(os.path.join(loss_path, loss_file), loss)

if __name__ == "__main__":
    args = parser.parse_args()
    flow.env.grpc_use_no_signal()
    flow.env.log_dir("./log")
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
                flow.deprecated.init_worker(
                    scp_binary=True, use_uuid=False)
            elif args.skip_scp_binary:
                flow.deprecated.init_worker(
                    scp_binary=False, use_uuid=False)
            else:
                flow.deprecated.init_worker(
                    scp_binary=True, use_uuid=True)

    main(args)
    if (args.multinode and args.skip_scp_binary is False and
          args.scp_binary_without_uuid is False):
        flow.deprecated.delete_worker()
