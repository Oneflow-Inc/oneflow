import oneflow as of
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from datetime import datetime


def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="channels_first",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=False,
    weight_initializer=of.random_uniform_initializer(),
    bias_initializer=None,
):
    weight_shape = (filters, input.static_shape[1], kernel_size, kernel_size)
    weight = of.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
    )
    output = of.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate
    )
    if use_bias:
        bias = of.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
        )
        output = of.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = of.keras.activations.relu(output)
        else:
            raise NotImplementedError

    return output


def _fully_connected_layer(
    name,
    input,
    units,
    activation=op_conf_util.kRelu,
    use_bias=False,
    kernel_initializer=None,
    bias_initializer=None,
    trainable=True,
):
    if kernel_initializer is None:
        kernel_initializer = op_conf_util.InitializerConf()
        kernel_initializer.truncated_normal_conf.std = 0.816496580927726

    output = of.layers.dense(
        input,
        units,
        activation,
        use_bias,
        kernel_initializer,
        bias_initializer,
        trainable,
        name,
    )

    if use_bias:
        bias = of.get_variable(
            name + "-bias",
            shape=(units,),
            dtype=input.dtype,
            initializer=bias_initializer,
        )
        output = of.nn.bias_add(output, bias)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = of.keras.activations.relu(output)
        else:
            raise NotImplementedError

    return output


def _data_load_layer(data_dir="/dataset/imagenet_227/train/32"):
    image_blob_conf = of.data.BlobConf(
        "encoded",
        shape=(227, 227, 3),
        dtype=of.float,
        codec=of.data.ImageCodec([of.data.ImagePreprocessor("bgr2rgb")]),
        preprocessors=[
            of.data.NormByChannelPreprocessor((123.68, 116.78, 103.94))
        ],
    )

    label_blob_conf = of.data.BlobConf(
        "class/label", shape=(), dtype=of.int32, codec=of.data.RawCodec()
    )

    return of.data.decode_ofrecord(data_dir, (image_blob_conf, label_blob_conf))


def alexnet_train(images, labels):
    conv1 = _conv2d_layer(
        "conv1", images, filters=64, kernel_size=11, strides=4, padding="VALID"
    )

    pool1 = of.nn.avg_pool2d(conv1, 3, 2, "VALID", "NCHW", name="pool1")

    conv2 = _conv2d_layer("conv2", pool1, filters=192, kernel_size=5)

    pool2 = of.nn.avg_pool2d(conv2, 3, 2, "VALID", "NCHW", name="pool2")

    conv3 = _conv2d_layer("conv3", pool2, filters=384)

    conv4 = _conv2d_layer("conv4", conv3, filters=384)

    conv5 = _conv2d_layer("conv5", conv4, filters=256)

    pool5 = of.nn.avg_pool2d(conv5, 3, 2, "VALID", "NCHW", name="pool5")

    fc1 = _fully_connected_layer("fc1", pool5, 4096)

    dropout1 = fc1

    fc2 = _fully_connected_layer("fc2", dropout1, 4096)

    dropout2 = fc2

    fc3 = _fully_connected_layer("fc3", dropout2, 1001)

    loss = of.nn.sparse_softmax_cross_entropy_with_logits(
        labels, fc3, name="softmax_loss"
    )

    return loss


def alexnet_evaluate():
    raise NotImplementedError


def alexnet_train_job():
    job_conf = of.get_cur_job_conf_builder()
    job_conf.batch_size(12).data_part_num(8).default_data_type(of.float)
    job_conf.train_conf()
    job_conf.train_conf().primary_lr = 0.00001
    job_conf.train_conf().num_of_batches_in_snapshot = 100
    job_conf.train_conf().model_update_conf.naive_conf.SetInParent()

    (images, labels) = _data_load_layer()
    loss = alexnet_train(images, labels)
    job_conf.get_train_conf_builder().add_loss(loss)
    return loss


if __name__ == "__main__":
    config = of.ConfigProtoBuilder()
    config.gpu_device_num(1)
    config.grpc_use_no_signal()
    # config.model_load_snapshot_path(args.model_load_dir)
    config.model_save_snapshots_path(
        "./model_save-{}".format(
            str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        )
    )
    config.ctrl_port(9727)
    # if args.multinode:
    #     config.ctrl_port(12138)
    #     config.machine([{'addr': '192.168.1.15'}, {'addr': '192.168.1.16'}])
    #     if args.remote_by_hand is False:
    #         if args.scp_binary_without_uuid:
    #             flow.deprecated.init_worker(
    #                 config, scp_binary=True, use_uuid=False)
    #         elif args.skip_scp_binary:
    #             flow.deprecated.init_worker(
    #                 config, scp_binary=False, use_uuid=False)
    #         else:
    #             flow.deprecated.init_worker(
    #                 config, scp_binary=True, use_uuid=True)
    of.init(config)

    of.add_job(alexnet_train_job)

    with of.Session() as sess:
        check_point = of.train.CheckPoint()
        check_point.restore().initialize_or_restore(session=sess)
        fmt_str = "{:>12}  {:>12}  {:>12.10f}"
        print(
            "{:>12}  {:>12}  {:>12}".format("iter", "loss type", "loss value")
        )
        for i in range(10):
            print(
                fmt_str.format(
                    i, "train loss:", sess.run(alexnet_train_job).get().mean()
                )
            )
            # if (i + 1) % 10 == 0:
            #     print(
            #         fmt_str.format(
            #             i, "eval loss:", sess.run(EvaluateAlexNet).get().mean()
            #         )
            #     )
            if (i + 1) % 100 == 0:
                check_point.save(session=sess)
        # if (
        #     args.multinode
        #     and args.skip_scp_binary is False
        #     and args.scp_binary_without_uuid is False
        # ):
        #     of.deprecated.delete_worker(config)
