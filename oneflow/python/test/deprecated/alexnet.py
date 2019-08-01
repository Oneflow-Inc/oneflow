import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import sys
import os
sys.path.insert(
    0, '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-2]))


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


def ImgClassifyDecoder(dlnet, data_dir=''):
    return dlnet.DecodeOFRecord(data_dir, name='decode', blob=[
        {
            'name': 'encoded',
            'shape': {'dim': [224, 224, 3]},
            'data_type': flow.float,
            'encode_case': {
                'jpeg': {
                    'preprocess': [{
                        'resize': {
                            'width': 224,
                            'height': 224,
                        },
                    }, ]
                },
            },
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
    job_conf.batch_size(512).data_part_num(1).default_data_type(flow.float)

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

    pool1 = dlnet.MaxPooling2D(
        conv1,
        name='pool1',
        pool_size=[3, 3],
        strides=[2, 2],
        **_ALEXNET_BASIC_POOLING_CONF)

    conv2 = dlnet.Conv2D(
        pool1,
        name='conv2',
        filters=192,
        kernel_size=[5, 5],
        strides=[1, 1],
        padding='same',
        **_ALEXNET_BASIC_CONV_CONF)

    pool2 = dlnet.MaxPooling2D(
        conv2,
        name='pool2',
        pool_size=[3, 3],
        strides=[2, 2],
        **_ALEXNET_BASIC_POOLING_CONF)

    conv3 = dlnet.Conv2D(
        pool2,
        name='conv3',
        filters=384,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding='same',
        **_ALEXNET_BASIC_CONV_CONF)

    conv4 = dlnet.Conv2D(
        conv3,
        name='conv4',
        filters=384,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding='same',
        **_ALEXNET_BASIC_CONV_CONF)

    conv5 = dlnet.Conv2D(
        conv4,
        name='conv5',
        filters=256,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding='same',
        **_ALEXNET_BASIC_CONV_CONF)

    pool5 = dlnet.MaxPooling2D(
        conv5,
        name='pool5',
        pool_size=[3, 3],
        strides=[2, 2],
        **_ALEXNET_BASIC_POOLING_CONF)

    fc1 = dlnet.FullyConnected(
        pool5,
        name='fc1',
        units=4096,
        activation=op_conf_util.kRelu,
        **_BASIC_FC_CONF)

    dropout1 = dlnet.Dropout(
        fc1,
        name='dropout1',
        rate=0.5)

    fc2 = dlnet.FullyConnected(
        dropout1,
        name='fc2',
        units=4096,
        activation=op_conf_util.kRelu,
        **_BASIC_FC_CONF)

    dropout2 = dlnet.Dropout(fc2, name='dropout2', rate=0.5)

    fc3 = dlnet.FullyConnected(
        dropout2,
        name='fc3',
        units=1001,
        activation=op_conf_util.kNone,
        **_BASIC_FC_CONF)

    softmax = dlnet.Softmax(
        fc3, name='softmax')
    softmax_loss = dlnet.SparseCrossEntropy(
        softmax, label_blob, name='softmax_loss')
    return softmax_loss


if __name__ == '__main__':

    is_training = False
    if len(sys.argv) > 1 and sys.argv[1] == 'is_train':
        is_training = True
    flow.add_job(AlexNet)
    with flow.Session() as sess:
        check_point = flow.train.CheckPoint()
        check_point.restore().initialize_or_restore(session=sess)
        for i in range(10):
            print(sess.run(AlexNet).get())
        check_point.save(session=sess)
