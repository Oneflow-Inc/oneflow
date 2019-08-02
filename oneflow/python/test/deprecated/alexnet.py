import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
config.grpc_use_no_signal()
flow.init(config)

_ALEXNET_BASIC_CONV_CONF = dict(
    data_format='channels_first',
    dilation_rate=[1, 1],
    use_bias=False,
    activation=op_conf_util.kRelu,
    bias_initializer=dict(
        constant_conf=dict(
            value=0.0)),
    weight_initializer=dict(
        truncated_normal_conf=dict(
        ))
)

_ALEXNET_BASIC_POOLING_CONF = {
    'padding': 'valid',
    'data_format': 'channels_first',
}

_BASIC_FC_CONF = dict(
    # use_bias=True,
    weight_initializer=dict(
        truncated_normal_conf=dict(
            std=0.816496580927726)),
    bias_initializer=dict(
        constant_conf=dict(
            value=0.0))
)


def Reshape(in_blob, shape):
    dl_net = in_blob.dl_net()
    return dl_net.Reshape(
        in_blob,
        shape={'dim': shape},
        has_dim0_in_shape=True)


def ImgClassifyDecoder(dlnet, data_dir=''):
    return dlnet.DecodeOFRecord(data_dir, name='decode', blob=[
        {
            'name': 'encoded',
            'shape': {'dim': [227, 227, 3]},
            'data_type': flow.float,
            'encode_case': {
                'jpeg': {
                    'preprocess': [{
                        'resize': {
                            'width': 227,
                            'height': 227,
                        },
                    }, ]
                },
            },
            # 'preprocess': [{
            #     'norm_by_channel_conf': {
            #         'mean_value': [123.68,
            #                        116.78,
            #                        103.94, ],
            #         'data_format': "channels_last",
            #     }
            # }],
        },
        {
            'name': 'class/label',
            'shape': {},
            'data_type': flow.int32,
            'encode_case': {'raw': {}},
        }
    ])


def AlexNet():
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(64).data_part_num(1).default_data_type(flow.float)
    # job_conf.train_conf()
    # job_conf.train_conf().primary_lr = 0.005
    # job_conf.train_conf().num_of_batches_in_snapshot = 100
    # job_conf.train_conf().model_update_conf.naive_conf.SetInParent()
    dlnet = flow.deprecated.get_cur_job_dlnet_builder()

    decoders = ImgClassifyDecoder(
        dlnet, data_dir='/dataset/imagenet_227/train/32')
    img_blob = decoders['encoded']
    transposers = dlnet.Transpose(
        img_blob, name='transpose', perm=[0, 3, 1, 2])

    # input blobs
    label_blob = decoders['class/label']
    img_blob = transposers

    conv1 = dlnet.Conv2D(
        img_blob,
        name='conv1',
        filters=64,
        kernel_size=[11, 11],
        strides=[4, 4],
        padding='valid',
        **_ALEXNET_BASIC_CONV_CONF)

    # pool1 = dlnet.MaxPooling2D(
    #     conv1,
    #     name='pool1',
    #     pool_size=[3, 3],
    #     strides=[2, 2],
    #     **_ALEXNET_BASIC_POOLING_CONF)

    # conv2 = dlnet.Conv2D(
    #     pool1,
    #     name='conv2',
    #     filters=192,
    #     kernel_size=[5, 5],
    #     strides=[1, 1],
    #     padding='same',
    #     **_ALEXNET_BASIC_CONV_CONF)

    # pool2 = dlnet.MaxPooling2D(
    #     conv2,
    #     name='pool2',
    #     pool_size=[3, 3],
    #     strides=[2, 2],
    #     **_ALEXNET_BASIC_POOLING_CONF)

    # conv3 = dlnet.Conv2D(
    #     pool2,
    #     name='conv3',
    #     filters=384,
    #     kernel_size=[3, 3],
    #     strides=[1, 1],
    #     padding='same',
    #     **_ALEXNET_BASIC_CONV_CONF)

    # conv4 = dlnet.Conv2D(
    #     conv3,
    #     name='conv4',
    #     filters=384,
    #     kernel_size=[3, 3],
    #     strides=[1, 1],
    #     padding='same',
    #     **_ALEXNET_BASIC_CONV_CONF)

    # conv5 = dlnet.Conv2D(
    #     conv4,
    #     name='conv5',
    #     filters=256,
    #     kernel_size=[3, 3],
    #     strides=[1, 1],
    #     padding='same',
    #     **_ALEXNET_BASIC_CONV_CONF)

    # pool5 = dlnet.MaxPooling2D(
    #     conv5,
    #     name='pool5',
    #     pool_size=[3, 3],
    #     strides=[2, 2],
    #     **_ALEXNET_BASIC_POOLING_CONF)

    # reshape = Reshape(
    #     pool5,
    #     [-1, 9216])

    # fc1 = dlnet.FullyConnected(
    #     reshape,
    #     name='fc1',
    #     units=4096,
    #     **_BASIC_FC_CONF)
    
    # relu1 = dlnet.Relu(fc1, name='relu1')

    # dropout1 = dlnet.Dropout(
    #     relu1,
    #     name='dropout1',
    #     rate=0.5)

    # fc2 = dlnet.FullyConnected(
    #     dropout1,
    #     name='fc2',
    #     units=4096,
    #     **_BASIC_FC_CONF)

    # relu2 = dlnet.Relu(fc2, name='relu2')

    # dropout2 = dlnet.Dropout(relu2, name='dropout2', rate=0.5)

    # fc3 = dlnet.FullyConnected(
    #     dropout2,
    #     name='fc3',
    #     units=1001,
    #     **_BASIC_FC_CONF)

    # softmax = dlnet.Softmax(
    #     fc3, name='softmax')
    # softmax_loss = dlnet.SparseCrossEntropy(
    #     softmax, label_blob, name='softmax_loss')
    # job_conf.train_conf().loss_lbn.extend(["softmax_loss/out"])
    return conv1


if __name__ == '__main__':
    flow.add_job(AlexNet)
    with flow.Session() as sess:
        check_point = flow.train.CheckPoint()
        check_point.restore().initialize_or_restore(session=sess)
        sess.sync()
        for i in range(10):
            fetched = sess.run(AlexNet).get()
            import numpy as np
            # print(fetched.sum())
            # print(np.max(np.abs(fetched)))
            if i is 0:
                # np.save("img_blob", fetched)
                print(fetched)
                np.save("conv1-fetched", fetched)
        # check_point.save(session=sess)
