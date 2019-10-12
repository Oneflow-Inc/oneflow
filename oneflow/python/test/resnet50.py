import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import argparse

# import hook

from datetime import datetime


# DATA_DIR = "/dataset/PNGS/PNG228/of_record"
DATA_DIR = "/dataset/PNGS/PNG228/of_record_repeated"
MODEL_LOAD = "/dataset/PNGS/cnns_model_for_test/resnet50/models/of_model"
MODEL_SAVE = "./output/model_save-{}".format(
    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
)
IMAGE_SIZE = 228
BLOCK_COUNTS = [3, 4, 6, 3]
BLOCK_FILTERS = [256, 512, 1024, 2048]
BLOCK_FILTERS_INNER = [64, 128, 256, 512]


parser = argparse.ArgumentParser()
parser.add_argument(
    "-g", "--gpu_num_per_node", type=int, default=1, required=False
)
parser.add_argument(
    "-m", "--multinode", default=False, action="store_true", required=False
)
parser.add_argument(
    "-s",
    "--skip_scp_binary",
    default=False,
    action="store_true",
    required=False,
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
parser.add_argument("-i", "--iter_num", type=int, default=10, required=False)
parser.add_argument(
    "-e", "--eval_dir", type=str, default=DATA_DIR, required=False
)
parser.add_argument(
    "-t", "--train_dir", type=str, default=DATA_DIR, required=False
)
parser.add_argument(
    "-load", "--model_load_dir", type=str, default=MODEL_LOAD, required=False
)
parser.add_argument(
    "-save", "--model_save_dir", type=str, default=MODEL_SAVE, required=False
)
parser.add_argument(
    "-dn", "--data_part_num", type=int, default=32, required=False
)
parser.add_argument(
    "-b", "--batch_size", type=int, default=8, required=False
)

g_args = parser.parse_args()
g_output = []
g_output_key = []
g_trainable = True


def _data_load(data_dir):
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

    return flow.data.decode_ofrecord(
        data_dir, (label_blob_conf, image_blob_conf),
        batch_size=g_args.batch_size, data_part_num=g_args.data_part_num, name="decode",
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


def resnet50(data_dir):
    (labels, images) = _data_load(data_dir)
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


@flow.function
def TrainNet():
    flow.config.train.primary_lr(0.0032)
    flow.config.train.model_update_conf(dict(naive_conf={}))

    _set_trainable(True)
    loss = resnet50(g_args.train_dir)
    flow.losses.add_loss(loss)
    return loss


@flow.function
def evaluate():
    _set_trainable(False)
    return resnet50(g_args.eval_dir)


def main():
    flow.config.default_data_type(flow.float)
    flow.config.gpu_device_num(g_args.gpu_num_per_node)
    flow.config.grpc_use_no_signal()
    flow.config.log_dir("./output/log")
    flow.config.ctrl_port(12138)

    if g_args.multinode:
        flow.config.ctrl_port(12139)
        flow.config.machine(
            [{"addr": "192.168.1.15"}, {"addr": "192.168.1.16"}]
        )

        if g_args.scp_binary_without_uuid:
            flow.deprecated.init_worker(scp_binary=True, use_uuid=False)
        elif g_args.skip_scp_binary:
            flow.deprecated.init_worker(scp_binary=False, use_uuid=False)
        else:
            flow.deprecated.init_worker(scp_binary=True, use_uuid=True)

    check_point = flow.train.CheckPoint()
    check_point.load(MODEL_LOAD)
    # if not g_args.model_load_dir:
    #     check_point.init()
    # else:
    #     check_point.load(g_args.model_load_dir)

    fmt_str = "{:>12}  {:>12}  {:.11f}"
    print("{:>12}  {:>12}  {:>12}".format("iter", "loss type", "loss value"))
    for i in range(g_args.iter_num):
        loss = TrainNet().get().mean()
        print(fmt_str.format(i, "train loss:", loss))

        # output_dict = dict(zip(g_output_key, g_output))
        # hook.dump_tensor_to_file(output_dict, "./prob_output/iter_{}".format(i))

        if (i + 1) % 100 == 0:
            eval = evaluate().get().mean()
            print(fmt_str.format(i, "eval loss:", eval))

            check_point.save(MODEL_SAVE + "_" + str(i))


if __name__ == "__main__":
    main()
