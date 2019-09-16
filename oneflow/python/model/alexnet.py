import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from datetime import datetime
import argparse

_DATA_DIR = "/dataset/imagenet_227/train/32"
_MODEL_SAVE_DIR = "./model_save-{}".format(
    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
)

parser = argparse.ArgumentParser(description="flags for multi-node and resource")
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-i", "--iter_num", type=int, default=10, required=False)
parser.add_argument(
    "-m", "--multinode", default=False, action="store_true", required=False
)
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
parser.add_argument("-load", "--model_load_dir", type=str, default="", required=False)
parser.add_argument(
    "-save", "--model_save_dir", type=str, default=_MODEL_SAVE_DIR, required=False
)

args = parser.parse_args()


def _conv2d_layer(
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
        if activation == op_conf_util.kRelu:
            output = flow.keras.activations.relu(output)
        else:
            raise NotImplementedError

    return output


def _data_load_layer(data_dir):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(227, 227, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec([flow.data.ImagePreprocessor("bgr2rgb")]),
        preprocessors=[flow.data.NormByChannelPreprocessor((123.68, 116.78, 103.94))],
    )

    label_blob_conf = flow.data.BlobConf(
        "class/label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        data_dir, (label_blob_conf, image_blob_conf),
        batch_size=12, data_part_num=8, name="decode"
    )


def alexnet(images, labels, trainable=True):
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

    def _get_initializer():
        kernel_initializer = op_conf_util.InitializerConf()
        kernel_initializer.truncated_normal_conf.std = 0.816496580927726
        return kernel_initializer

    if len(pool5.shape) > 2:
        pool5 = flow.reshape(pool5, shape=(pool5.static_shape[0], -1))

    fc1 = flow.layers.dense(
        inputs=pool5,
        units=4096,
        activation=flow.keras.activations.relu,
        use_bias=False,
        kernel_initializer=_get_initializer(),
        bias_initializer=False,
        trainable=trainable,
        name="fc1",
    )

    dropout1 = fc1

    fc2 = flow.layers.dense(
        inputs=dropout1,
        units=4096,
        activation=flow.keras.activations.relu,
        use_bias=False,
        kernel_initializer=_get_initializer(),
        bias_initializer=False,
        trainable=trainable,
        name="fc2",
    )

    dropout2 = fc2

    fc3 = flow.layers.dense(
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

@flow.function
def alexnet_train_job():
    flow.config.train.primary_lr(0.00001)
    flow.config.train.model_update_conf(dict(naive_conf={}))

    (labels, images) = _data_load_layer(args.train_dir)
    loss = alexnet(images, labels)
    flow.losses.add_loss(loss)
    return loss


@flow.function
def alexnet_eval_job():
    (labels, images) = _data_load_layer(args.eval_dir)
    return alexnet(images, labels, False)


if __name__ == "__main__":
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.config.ctrl_port(9788)

    flow.config.default_data_type(flow.float)

    if args.multinode:
        flow.config.ctrl_port(12138)
        flow.config.machine([{"addr": "192.168.1.15"}, {"addr": "192.168.1.16"}])
        if args.remote_by_hand is False:
            if args.scp_binary_without_uuid:
                flow.deprecated.init_worker(scp_binary=True, use_uuid=False)
            elif args.skip_scp_binary:
                flow.deprecated.init_worker(scp_binary=False, use_uuid=False)
            else:
                flow.deprecated.init_worker(scp_binary=True, use_uuid=True)

    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)
    fmt_str = "{:>12}  {:>12}  {:>12.10f}"
    print("{:>12}  {:>12}  {:>12}".format("iter", "loss type", "loss value"))
    for i in range(10):
        print(
            fmt_str.format(
                i, "train loss:", alexnet_train_job().get().mean()
            )
        )
        if (i + 1) % 10 == 0:
            print(
                fmt_str.format(
                    i, "eval loss:", alexnet_eval_job().get().mean()
                )
            )
        if (i + 1) % 100 == 0:
            check_point.save(_MODEL_SAVE_DIR + str(i))
    if (
        args.multinode
        and args.skip_scp_binary is False
        and args.scp_binary_without_uuid is False
    ):
        flow.deprecated.delete_worker()
