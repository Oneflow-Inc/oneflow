# ###################################################################
# alexnet.py
# Usage:
#     Single Node: python alexnet.py -g 1
#               -g gpu number
#     Multi Nodes: python alexnet.py -g 8 -m -n "192.168.1.15,192.168.1.16"
#               -g gpu number
#               -m run on multi nodes
#               -n IP addresses of nodes, seperated by comma
# ###################################################################

import oneflow as flow
import argparse

DATA_DIR = "/dataset/imagenet_1k/oneflow/30/train"
parser = argparse.ArgumentParser(description="flags for multi-node and resource")
parser.add_argument("-i", "--iter_num", type=int, default=10, required=False)
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument(
    "-m", "--multinode", default=False, action="store_true", required=False
)
parser.add_argument("-n", "--node_list", type=str, default=None, required=False)
parser.add_argument("-e", "--eval_dir", type=str, default=DATA_DIR, required=False)
parser.add_argument("-t", "--train_dir", type=str, default=DATA_DIR, required=False)
parser.add_argument("-load", "--model_load_dir", type=str, default="", required=False)
parser.add_argument(
    "-save", "--model_save_dir", type=str, default="./checkpoints", required=False
)
args = parser.parse_args()


def _data_load_layer(data_dir):
    # load image from dataset and preprocess
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(227, 227, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec([flow.data.ImageResizePreprocessor(227, 227)]),
        preprocessors=[flow.data.NormByChannelPreprocessor((123.68, 116.78, 103.94))],
    )

    # load label from dataset
    label_blob_conf = flow.data.BlobConf(
        "class/label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    # decode
    labels, images = flow.data.decode_ofrecord(
        data_dir,
        (label_blob_conf, image_blob_conf),
        batch_size=12,
        data_part_num=8,
        name="decode",
    )

    return labels, images


def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation="Relu",
    use_bias=False,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=None,
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
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == "Relu":
            output = flow.keras.activations.relu(output)
        else:
            raise NotImplementedError

    return output


def alexnet(images, labels):
    # transpose, NHWC -> NCHW
    transposed = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])

    conv1 = _conv2d_layer(
        "conv1", transposed, filters=64, kernel_size=11, strides=4, padding="VALID"
    )

    pool1 = flow.nn.avg_pool2d(conv1, 3, 2, "VALID", "NCHW", name="pool1")

    conv2 = _conv2d_layer("conv2", pool1, filters=192, kernel_size=5)

    pool2 = flow.nn.avg_pool2d(conv2, 3, 2, "VALID", "NCHW", name="pool2")

    conv3 = _conv2d_layer("conv3", pool2, filters=384)

    conv4 = _conv2d_layer("conv4", conv3, filters=384)

    conv5 = _conv2d_layer("conv5", conv4, filters=256)

    pool5 = flow.nn.avg_pool2d(conv5, 3, 2, "VALID", "NCHW", name="pool5")

    if len(pool5.shape) > 2:
        pool5 = flow.reshape(pool5, shape=(pool5.static_shape[0], -1))

    fc1 = flow.layers.dense(
        inputs=pool5,
        units=4096,
        activation=flow.keras.activations.relu,
        use_bias=False,
        kernel_initializer=flow.random_uniform_initializer(),
        bias_initializer=False,
        trainable=True,
        name="fc1",
    )

    dropout1 = flow.nn.dropout(fc1, rate=0.5)

    fc2 = flow.layers.dense(
        inputs=dropout1,
        units=4096,
        activation=flow.keras.activations.relu,
        use_bias=False,
        kernel_initializer=flow.random_uniform_initializer(),
        bias_initializer=False,
        trainable=True,
        name="fc2",
    )

    dropout2 = flow.nn.dropout(fc2, rate=0.5)

    fc3 = flow.layers.dense(
        inputs=dropout2,
        units=1001,
        activation=None,
        use_bias=False,
        kernel_initializer=flow.random_uniform_initializer(),
        bias_initializer=False,
        trainable=True,
        name="fc3",
    )

    # loss function
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, fc3, name="softmax_loss"
    )

    return loss


# train job
@flow.function
def alexnet_train_job():
    # set hyper parameter
    flow.config.train.primary_lr(0.00001)
    flow.config.train.model_update_conf(dict(naive_conf={}))

    # load data
    (labels, images) = _data_load_layer(args.train_dir)

    # construct network
    loss = alexnet(images, labels)

    # set loss
    flow.losses.add_loss(loss)

    return loss


# inference job
@flow.function
def alexnet_eval_job():
    # load data
    (labels, images) = _data_load_layer(args.eval_dir)

    # construct inference network
    loss = alexnet(images, labels)

    return loss


def main():
    # set running mode
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.config.ctrl_port(9788)
    flow.config.default_data_type(flow.float)

    # set multi nodes mode port
    if args.multinode:
        flow.config.ctrl_port(12138)
        nodes = []
        for n in args.node_list.strip().split(","):
            addr_dict = {}
            addr_dict["addr"] = n
            nodes.append(addr_dict)
        flow.config.machine(nodes)

    # load/initialize model
    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)

    # training iter 
    print("{:>12}  {:>12}  {:>12}".format("iter", "loss type", "loss value"))
    for i in range(args.iter_num):
        fmt_str = "{:>12}  {:>12}  {:>12.10f}"

        # print training log
        train_loss = alexnet_train_job().get().mean()
        print(fmt_str.format(i, "train loss:", train_loss))

        # print inference log
        if (i + 1) % 10 == 0:
            eval_loss = alexnet_eval_job().get().mean()
            print(fmt_str.format(i, "eval loss:", eval_loss))

        # save model
        if (i + 1) % 100 == 0:
            check_point.save(args.model_save_dir + str(i))


if __name__ == "__main__":
    main()
