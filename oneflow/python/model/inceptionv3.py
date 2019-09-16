import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from datetime import datetime
import argparse

_DATA_DIR = "/dataset/PNGS/PNG299/of_record_repeated"
_EVAL_DIR = _DATA_DIR
_TRAIN_DIR = _DATA_DIR
_MODEL_LOAD = "/dataset/PNGS/cnns_model_for_test/inceptionv3/models/of_model"
_MODEL_SAVE_DIR = "./model_save-{}".format(
    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
)

parser = argparse.ArgumentParser(description="flags for multi-node and resource")
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-b", "--batch_size", type=int, default=2, required=False)
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
parser.add_argument(
    "-load", "--model_load_dir", type=str, default=_MODEL_LOAD, required=False
)
parser.add_argument(
    "-save", "--model_save_dir", type=str, default=_MODEL_SAVE_DIR, required=False
)

args = parser.parse_args()

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
    weight_shape = (filters, input.static_shape[1]) + kernel_size
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
        elif activation == op_conf_util.kSigmoid:
            output = flow.keras.activations.sigmoid(output)
        else:
            raise NotImplementedError

    return output


def _data_load_layer(data_dir):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(299, 299, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec([flow.data.ImagePreprocessor("bgr2rgb")]),
        preprocessors=[flow.data.NormByChannelPreprocessor((123.68, 116.78, 103.94))],
    )
    label_blob_conf = flow.data.BlobConf(
        "class/label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )
    return flow.data.decode_ofrecord(
        data_dir, (image_blob_conf, label_blob_conf),
        batch_size=args.batch_size, data_part_num=32, name="decode"
    )


def InceptionA(in_blob, index):
    with flow.deprecated.variable_scope("mixed_{}".format(index)):
        with flow.deprecated.variable_scope("branch1x1"):
            branch1x1 = _conv2d_layer(
                "conv0", in_blob, filters=64, kernel_size=1, strides=1, padding="SAME"
            )
        with flow.deprecated.variable_scope("branch5x5"):
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
        with flow.deprecated.variable_scope("branch3x3dbl"):
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
        with flow.deprecated.variable_scope("branch_pool"):
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
    with flow.deprecated.variable_scope("mixed_{}".format(index)):
        with flow.deprecated.variable_scope("branch3x3"):
            branch3x3 = _conv2d_layer(
                "conv0", in_blob, filters=384, kernel_size=3, strides=2, padding="VALID"
            )
        with flow.deprecated.variable_scope("branch3x3dbl"):
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
        with flow.deprecated.variable_scope("branch_pool"):
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
    with flow.deprecated.variable_scope("mixed_{}".format(index)):
        with flow.deprecated.variable_scope("branch1x1"):
            branch1x1 = _conv2d_layer(
                "conv0", in_blob, filters=192, kernel_size=1, strides=1, padding="SAME"
            )
        with flow.deprecated.variable_scope("branch7x7"):
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
        with flow.deprecated.variable_scope("branch7x7dbl"):
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
        with flow.deprecated.variable_scope("branch_pool"):
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
    with flow.deprecated.variable_scope("mixed_{}".format(index)):
        with flow.deprecated.variable_scope("branch3x3"):
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
        with flow.deprecated.variable_scope("branch7x7x3"):
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
        with flow.deprecated.variable_scope("branch_pool"):
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
    with flow.deprecated.variable_scope("mixed_{}".format(index)):
        with flow.deprecated.variable_scope("branch1x1"):
            branch1x1 = _conv2d_layer(
                "conv0", in_blob, filters=320, kernel_size=1, strides=1, padding="SAME"
            )
        with flow.deprecated.variable_scope("branch3x3"):
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
        with flow.deprecated.variable_scope("branch3x3dbl"):
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
        with flow.deprecated.variable_scope("branch_pool"):
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
    images = flow.transpose(images, perm=[0, 3, 1, 2])

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

    with flow.deprecated.variable_scope("logits"):
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


@flow.function
def TrainNet():
    flow.config.train.primary_lr(0.0001)
    flow.config.train.model_update_conf(dict(naive_conf={}))

    (images, labels) = _data_load_layer(args.train_dir)
    loss = InceptionV3(images, labels)
    flow.losses.add_loss(loss)
    return loss


if __name__ == "__main__":
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.config.ctrl_port(9678)
    flow.config.default_data_type(flow.float)

    if args.multinode:
        flow.config.ctrl_port(8975)
        flow.machine([{"addr": "192.168.1.11"}, {"addr": "192.168.1.12"}])
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
        print(fmt_str.format(i, "train loss:", TrainNet().get()[0].mean()))
        if (i + 1) % 100 == 0:
            check_point.save(_MODEL_SAVE_DIR + str(i))

    if (
        args.multinode
        and args.skip_scp_binary is False
        and args.scp_binary_without_uuid is False
    ):
        flow.deprecated.delete_worker()
