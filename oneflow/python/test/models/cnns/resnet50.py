import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import argparse
import os
import numpy
from datetime import datetime
# import hook

# DATA_DIR = "/dataset/PNGS/PNG228/of_record"
DATA_DIR = "/dataset/PNGS/PNG228/of_record_repeated"
MODEL_LOAD = "/dataset/PNGS/cnns_model_for_test/resnet50/models/of_model"
MODEL_SAVE = "./output/model_save-{}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))
NODE_LIST = "192.168.1.12,192.168.1.14"

IMAGE_SIZE = 228
BLOCK_COUNTS = [3, 4, 6, 3]
BLOCK_FILTERS = [256, 512, 1024, 2048]
BLOCK_FILTERS_INNER = [64, 128, 256, 512]

class DLNetSpec(object):
  def __init__(self):
    self.batch_size = 8
    self.data_part_num = 32
    self.eval_dir = DATA_DIR
    self.train_dir = DATA_DIR
    self.model_save_dir = MODEL_SAVE
    self.model_load_dir = MODEL_LOAD
    self.num_nodes = 1
    self.gpu_num_per_node = 1
    self.iter_num = 10

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-i", "--iter_num", type=int, default=10, required=False)
parser.add_argument("-m", "--multinode", default=False, action="store_true", required=False)
parser.add_argument("-n", "--node_list", type=str, default=NODE_LIST, required=False)
parser.add_argument("-s", "--skip_scp_binary", default=False, action="store_true", required=False)
parser.add_argument("-c","--scp_binary_without_uuid",default=False,action="store_true",required=False)
parser.add_argument("-r", "--remote_by_hand", default=False, action="store_true", required=False)
parser.add_argument("-e", "--eval_dir", type=str, default=DATA_DIR, required=False)
parser.add_argument("-t", "--train_dir", type=str, default=DATA_DIR, required=False)
parser.add_argument("-load", "--model_load_dir", type=str, default=MODEL_LOAD, required=False)
parser.add_argument("-save", "--model_save_dir", type=str, default=MODEL_SAVE, required=False)
parser.add_argument("-dn", "--data_part_num", type=int, default=32, required=False)
parser.add_argument("-b", "--batch_size", type=int, default=8, required=False)

g_output = []
g_output_key = []
g_trainable = True


def _data_load(args, data_dir):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec([flow.data.ImagePreprocessor("bgr2rgb")]),
        preprocessors=[
            flow.data.NormByChannelPreprocessor((123.68, 116.78, 103.94))
        ],
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


def _conv2d(
    name,
    input,
    filters,
    kernel_size,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilations=1,
    weight_initializer=flow.variance_scaling_initializer(),
):
    weight = flow.get_variable(
        name + "-weight",
        shape=(filters, input.static_shape[1], kernel_size, kernel_size),
        dtype=input.dtype,
        initializer=weight_initializer,
        trainable=g_trainable,
    )
    return flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilations, name=name
    )


def _batch_norm(inputs, name=None):
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=1,
        momentum=0.997,
        epsilon=1e-5,
        center=True,
        scale=True,
        trainable=g_trainable,
        name=name,
    )


def conv2d_affine(
    input, name, filters, kernel_size, strides, activation=op_conf_util.kNone
):
    # input data_format must be NCHW, cannot check now
    padding = "SAME" if strides > 1 or kernel_size > 1 else "VALID"
    output = _conv2d(name, input, filters, kernel_size, strides, padding)
    # output = _batch_norm(output, name + "_bn")
    # if activation != op_conf_util.kNone:
    #     output = flow.keras.activations.relu(output)

    return output


def bottleneck_transformation(
    input, block_name, filters, filters_inner, strides
):
    a = conv2d_affine(
        input,
        block_name + "_branch2a",
        filters_inner,
        1,
        1,
        activation=op_conf_util.kRelu,
    )

    b = conv2d_affine(
        a,
        block_name + "_branch2b",
        filters_inner,
        1,  # 1 for test origin 3
        strides,
        activation=op_conf_util.kRelu,
    )

    c = conv2d_affine(b, block_name + "_branch2c", filters, 1, 1)

    return c


def residual_block(input, block_name, filters, filters_inner, strides_init):
    if strides_init != 1 or block_name == "res2_0":
        shortcut = conv2d_affine(
            input, block_name + "_branch1", filters, 1, strides_init
        )
    else:
        shortcut = input

    bottleneck = bottleneck_transformation(
        input, block_name, filters, filters_inner, strides_init
    )

    return flow.keras.activations.relu(shortcut + bottleneck)


def residual_stage(
    input, stage_name, counts, filters, filters_inner, stride_init=2
):
    output = input
    for i in range(counts):
        block_name = "%s_%d" % (stage_name, i)
        output = residual_block(
            output,
            block_name,
            filters,
            filters_inner,
            stride_init if i == 0 else 1,
        )

    return output


def resnet_conv_x_body(input, on_stage_end=lambda x: x):
    output = input
    for i, (counts, filters, filters_inner) in enumerate(
        zip(BLOCK_COUNTS, BLOCK_FILTERS, BLOCK_FILTERS_INNER)
    ):
        stage_name = "res%d" % (i + 2)
        output = residual_stage(
            output,
            stage_name,
            counts,
            filters,
            filters_inner,
            1 if i == 0 else 2,
        )
        on_stage_end(output)
        g_output_key.append(stage_name)
        g_output.append(output)

    return output


def resnet_stem(input):
    conv1 = _conv2d("conv1", input, 64, 7, 2)
    g_output_key.append("conv1")
    g_output.append(conv1)

    # conv1_bn = flow.keras.activations.relu(_batch_norm(conv1, "conv1_bn"))
    # for test
    conv1_bn = conv1

    pool1 = flow.nn.avg_pool2d(
        conv1_bn,
        ksize=3,
        strides=2,
        padding="VALID",
        data_format="NCHW",
        name="pool1",
    )
    g_output_key.append("pool1")
    g_output.append(pool1)

    return pool1


def resnet50(args, data_dir):
    (labels, images) = _data_load(args, data_dir)
    images = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])
    g_output_key.append("input_img")
    g_output.append(images)

    with flow.deprecated.variable_scope("Resnet"):
        stem = resnet_stem(images)
        body = resnet_conv_x_body(stem, lambda x: x)
        pool5 = flow.nn.avg_pool2d(
            body,
            ksize=7,
            strides=1,
            padding="VALID",
            data_format="NCHW",
            name="pool5",
        )
        g_output_key.append("pool5")
        g_output.append(pool5)

        fc1001 = flow.layers.dense(
            flow.reshape(pool5, (pool5.shape[0], -1)),
            units=1001,
            use_bias=True,
            kernel_initializer=flow.xavier_uniform_initializer(),
            bias_initializer=flow.zeros_initializer(),
            trainable=g_trainable,
            name="fc1001",
        )
        g_output_key.append("fc1001")
        g_output.append(fc1001)

        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, fc1001, name="softmax_loss"
        )
        g_output_key.append("cross_entropy")
        g_output.append(loss)

    return loss


def _set_trainable(trainable):
    global g_trainable
    g_trainable = trainable


def main(args):
    flow.config.machine_num(args.num_nodes)
    flow.config.gpu_device_num(args.gpu_num_per_node)

    train_config = flow.FunctionConfig()
    train_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    train_config.default_data_type(flow.float)
    train_config.train.primary_lr(0.0032)
    train_config.train.model_update_conf(dict(naive_conf={}))
    @flow.function(train_config)
    def TrainNet():
        _set_trainable(True)
        loss = resnet50(args, args.train_dir)
        flow.losses.add_loss(loss)
        return loss

    eval_config = flow.FunctionConfig()
    eval_config.default_data_type(flow.float)
    @flow.function(eval_config)
    def evaluate():
        with flow.distribute.consistent_strategy():
            _set_trainable(False)
            return resnet50(args, args.eval_dir)

    check_point = flow.train.CheckPoint()
    check_point.load(MODEL_LOAD)
    # if not args.model_load_dir:
    #     check_point.init()
    # else:
    #     check_point.load(args.model_load_dir)

    loss = []

    fmt_str = "{:>12}  {:>12}  {:.6f}"
    print("{:>12}  {:>12}  {:>12}".format("iter", "loss type", "loss value"))
    for i in range(args.iter_num):
        train_loss = TrainNet().get().mean()

        loss.append(train_loss)
        print(fmt_str.format(i, "train loss:", train_loss))

        # output_dict = dict(zip(g_output_key, g_output))
        # hook.dump_tensor_to_file(output_dict, "./prob_output/iter_{}".format(i))

        # if (i + 1) % 100 == 0:
        #     eval = evaluate().get().mean()
        #     print(fmt_str.format(i, "eval loss:", eval))

        #     check_point.save(MODEL_SAVE + "_" + str(i))
        
    # save loss to file
    loss_file = "{}n{}c.npy".format(str(args.num_nodes), str(args.gpu_num_per_node * args.num_nodes))
    loss_path = "./of_loss/resnet50"
    if not os.path.exists(loss_path): os.makedirs(loss_path)
    numpy.save(os.path.join(loss_path, loss_file), loss)


if __name__ == "__main__":
    flow.env.log_dir("./output/log")
    flow.env.ctrl_port(12138)
    args = parser.parse_args()
    if args.multinode:
        flow.env.ctrl_port(12139)
        nodes = []
        for n in args.node_list.strip().split(","):
            addr_dict = {}
            addr_dict["addr"] = n
            nodes.append(addr_dict)

        flow.env.machine(nodes)

        if args.scp_binary_without_uuid:
            flow.deprecated.init_worker(scp_binary=True, use_uuid=False)
        elif args.skip_scp_binary:
            flow.deprecated.init_worker(scp_binary=False, use_uuid=False)
        else:
            flow.deprecated.init_worker(scp_binary=True, use_uuid=True)
    num_nodes = len(args.node_list.strip().split(",")) if args.multinode else 1
    print("Traning resnet50: num_gpu_per_node = {}, num_nodes = {}.".format(args.gpu_num_per_node, num_nodes))
    main(args)
    if (args.multinode and args.skip_scp_binary is False
        and args.scp_binary_without_uuid is False):
      flow.deprecated.delete_worker()
