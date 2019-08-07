from __future__ import print_function
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import os
import shutil

config = flow.ConfigProtoBuilder()
config.gpu_device_num(1)
config.grpc_use_no_signal()
config.model_load_snapshot_path(
    "/home/caishenghang/dev/cnns_test/alexnet_fp16/fp32/of_model")
_MODEL_SAVE = "./model_save"
config.model_save_snapshots_path(_MODEL_SAVE)
flow.init(config)

if os.path.isdir(_MODEL_SAVE):
    shutil.rmtree(_MODEL_SAVE)

_DATA_DIR = "/dataset/imagenet_227/train/32"
_SINGLE_PIC_DATA_DIR = '/home/caishenghang/dev/cnns_test/dataset/PNG227/of_record'
_EVAL_DIR = _DATA_DIR
_TRAIN_DIR = _SINGLE_PIC_DATA_DIR

_ALEXNET_BASIC_CONV_CONF = dict(
    data_format='channels_first',
    dilation_rate=[1, 1],
    use_bias=False,
    bias_initializer=dict(
        constant_conf=dict(
            value=0.0)),
    weight_initializer=dict(
        random_uniform_conf=dict(
        ))
)

_ALEXNET_BASIC_POOLING_CONF = {
    'padding': 'valid',
    'data_format': 'channels_first',
}

_BASIC_FC_CONF = dict(
    use_bias=False,
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
            # 'encode_case': {
            #     'jpeg': {
            #         'preprocess': [{
            #             'resize': {
            #                 'width': 227,
            #                 'height': 227,
            #             },
            #         }, ]
            #     },
            # },
            'encode_case': {
                'jpeg': {
                    'preprocess': [{
                        'bgr2rgb': {
                        },
                    }, ]
                },
            },
            'preprocess': [{
                'norm_by_channel_conf': {
                    'mean_value': [123.68,
                                   116.78,
                                   103.94, ],
                    'data_format': "channels_last",
                }
            }],
        },
        {
            'name': 'class/label',
            'shape': {},
            'data_type': flow.int32,
            'encode_case': {'raw': {}},
        }
    ])


def BuildAlexNetWithDeprecatedAPI(data_dir):
    dlnet = flow.deprecated.get_cur_job_dlnet_builder()

    decoders = ImgClassifyDecoder(
        dlnet, data_dir=data_dir)
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

    conv1 = dlnet.Relu(conv1)

    pool1 = dlnet.AveragePooling2D(
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

    conv2 = dlnet.Relu(conv2)

    pool2 = dlnet.AveragePooling2D(
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

    conv3 = dlnet.Relu(conv3)

    conv4 = dlnet.Conv2D(
        conv3,
        name='conv4',
        filters=384,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding='same',
        **_ALEXNET_BASIC_CONV_CONF)

    conv4 = dlnet.Relu(conv4)

    conv5 = dlnet.Conv2D(
        conv4,
        name='conv5',
        filters=256,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding='same',
        **_ALEXNET_BASIC_CONV_CONF)

    conv5 = dlnet.Relu(conv5)

    pool5 = dlnet.AveragePooling2D(
        conv5,
        name='pool5',
        pool_size=[3, 3],
        strides=[2, 2],
        **_ALEXNET_BASIC_POOLING_CONF)

    reshape = Reshape(
        pool5,
        [-1, 9216])

    fc1 = dlnet.FullyConnected(
        reshape,
        name='fc1',
        units=4096,
        **_BASIC_FC_CONF)

    fc1 = dlnet.Relu(fc1)
    dropout1 = fc1
    # dropout1 = dlnet.Dropout(
    #     fc1,
    #     name='dropout1',
    #     rate=0.5)

    fc2 = dlnet.FullyConnected(
        dropout1,
        name='fc2',
        units=4096,
        **_BASIC_FC_CONF)

    fc2 = dlnet.Relu(fc2)
    dropout2 = fc2
    # dropout2 = dlnet.Dropout(fc2, name='dropout2', rate=0.5)

    fc3 = dlnet.FullyConnected(
        dropout2,
        name='fc3',
        units=1001,
        **_BASIC_FC_CONF)

    softmax = dlnet.Softmax(
        fc3, name='softmax')
    softmax_loss = dlnet.SparseCrossEntropy(
        softmax, label_blob, name='softmax_loss')
    return softmax_loss


def TrainAlexNet():
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
    job_conf.train_conf()
    job_conf.train_conf().primary_lr = 0.00001
    job_conf.train_conf().num_of_batches_in_snapshot = 100
    job_conf.train_conf().model_update_conf.naive_conf.SetInParent()
    job_conf.train_conf().loss_lbn.extend(["softmax_loss/out"])
    return BuildAlexNetWithDeprecatedAPI(_TRAIN_DIR)


def EvaluateAlexNet():
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(256).data_part_num(1).default_data_type(flow.float)
    return BuildAlexNetWithDeprecatedAPI(_EVAL_DIR)


flow.add_job(TrainAlexNet)
flow.add_job(EvaluateAlexNet)
if __name__ == '__main__':
    with flow.Session() as sess:
        check_point = flow.train.CheckPoint()
        check_point.restore().initialize_or_restore(session=sess)
        fmt_str = '{:>12}  {:>12}  {:>12.3f}'
        print('{:>12}  {:>12}  {:>12}'.format(
            "iter", "loss type", "loss value"))
        for i in range(100):
            print(fmt_str.format(i, "train loss:", sess.run(
                TrainAlexNet).get().mean()))
            if (i + 1) % 10 is 0:
                print(fmt_str.format(i, "eval loss:", sess.run(
                    EvaluateAlexNet).get().mean()))
            if (i + 1) % 100 is 0:
                check_point.save(session=sess)
