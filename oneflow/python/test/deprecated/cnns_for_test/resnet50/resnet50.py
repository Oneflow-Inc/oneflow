from __future__ import print_function
from datetime import datetime
import argparse
import sys

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util

from util import ExtendDict
import hook

# -------------------------------------------------------------------- #
# Building up model.
# -------------------------------------------------------------------- #
IMAGE_SIZE = 228

_RESNET_BASIC_CONF = {
    'trainable': True,
}

_RESNET_BASIC_CONV_CONF = ExtendDict(
    _RESNET_BASIC_CONF,
    padding='valid',
    data_format='channels_first',
    dilation_rate=[1, 1],
    weight_initializer=dict(
        random_uniform_conf=dict(
        )),
    use_bias=False,
)

_RESNET_BASIC_POOLING_CONF = {
    'padding': 'valid',
    'data_format': 'channels_first',
}

_BASIC_FC_CONF = dict(
    use_bias=True,
    weight_initializer=dict(
        truncated_normal_conf=dict(
            std=0.816496580927726)),
    bias_initializer=dict(
        constant_conf=dict(
            value=0.0))
)


def _FullyConnected(input_blob, name=None, input_size=1, units=1, use_bias=True, activation=None,
                    weight_initializer=None, bias_initializer=None):
    dl_net = input_blob.dl_net()

    weight_blob = dl_net.Variable(
        name=name + '-weight',
        shape={'dim': [input_size, units]},
        initializer=weight_initializer,
        model_name='weight')

    bias_blob = dl_net.Variable(
        name=name + '-bias',
        shape={'dim': [units]},
        initializer=bias_initializer,
        model_name='bias')

    output_blob = dl_net.Matmul(input_blob, weight_blob)  # , transpose_b=True)
    output_blob = dl_net.BiasAdd(output_blob, bias_blob)
    return output_blob


def Reshape(in_blob, shape):
    dl_net = in_blob.dl_net()
    return dl_net.Reshape(
        in_blob,
        shape={'dim': shape},
        has_dim0_in_shape=True)


def BasicStem(img_blob):
    dl_net = img_blob.dl_net()
    conv1 = dl_net.Conv2D(img_blob,
                          name='conv1',
                          filters=64,
                          kernel_size=[7, 7],
                          strides=[2, 2],
                          **_RESNET_BASIC_CONV_CONF)

    prob_list_key.append("conv1")
    prob_list.append(conv1)

    if _RESNET_BASIC_CONF['trainable']:
        # conv1_bn = dl_net.Normalization(conv1,
        #                             name='conv1_bn',
        #                             axis=1,
        #                             activation=op_conf_util.kRelu,
        #                             momentum=0.997,
        #                             epsilon=1e-5,
        #                             **_RESNET_BASIC_CONF);
        pool1 = dl_net.AveragePooling2D(conv1,
                                        name='pool1',
                                        pool_size=[3, 3],
                                        strides=[2, 2],
                                        **_RESNET_BASIC_POOLING_CONF)

        prob_list_key.append("pool1")
        prob_list.append(pool1)
    else:
        conv1_bn = dl_net.AffineChannel(conv1,
                                        name='conv1_bn',
                                        channel_axis=1,
                                        activation=op_conf_util.kRelu,
                                        **_RESNET_BASIC_CONF)

    # pool1 = dl_net.MaxPooling2D(conv1_bn,
    #                            name='pool1',
    #                            pool_size=[3, 3],
    #                            strides=[2, 2],
    #                            **_RESNET_BASIC_POOLING_CONF);

    return pool1


def Conv2DAffine(in_blob, name, filters, kernel_size, strides, activation=op_conf_util.kNone):
    dl_net = in_blob.dl_net()
    if type(kernel_size) not in [tuple, list]:
        kernel_size = [kernel_size, kernel_size]
    if type(strides) not in [tuple, list]:
        strides = [strides, strides]

    blob = dl_net.Conv2D(in_blob,
                         name=name,
                         filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         **_RESNET_BASIC_CONV_CONF)

    # conf = _RESNET_BASIC_CONF
    # if activation != op_conf_util.kNone:
    #     conf = ExtendDict(conf, activation=activation)
    # if _RESNET_BASIC_CONF['trainable']:
    #     return dl_net.Normalization(blob, name=name + "_bn", axis=1, **conf)
    # else:
    #     return dl_net.AffineChannel(blob, name=name + "_bn", channel_axis=1, **conf)
    return blob


def BottleneckTransformation(in_blob, prefix, filters, filters_inner, strides):
    dl_net = in_blob.dl_net()
    blob = Conv2DAffine(in_blob, prefix + '_branch2a', filters_inner,
                        1, strides, activation=op_conf_util.kRelu)

    blob = Conv2DAffine(blob, prefix + '_branch2b', filters_inner,
                        1, 1, activation=op_conf_util.kRelu)
    # for test, change 3*3 kernel to 1*1
    # blob = Conv2DAffine(blob, prefix + '_branch2b', filters_inner,
    #                    3, 1, activation=op_conf_util.kRelu)

    return Conv2DAffine(blob, prefix + '_branch2c', filters, 1, 1)


def ResidualBlock(in_blob, prefix, filters, filters_inner, strides_init):
    dl_net = in_blob.dl_net()
    shortcut_blob = in_blob
    if strides_init != 1 or prefix == 'res2_0':
        shortcut_blob = Conv2DAffine(
            shortcut_blob, prefix + '_branch1', filters, 1, strides_init)

    bottleneck_blob = BottleneckTransformation(
        in_blob, prefix, filters, filters_inner, strides_init)

    res = dl_net.Add((shortcut_blob, bottleneck_blob),
                     name=prefix + '_sum', activation=op_conf_util.kRelu)
    res = dl_net.Relu(res)

    return res


def ResidualStage(in_blob, prefix, n, filters, filters_inner, stride_init=2):
    blob = in_blob
    dl_net = blob.dl_net()
    for i in range(n):
        blob = ResidualBlock(blob, "%s_%d" % (prefix, i), filters,
                             filters_inner, stride_init if i == 0 else 1)
    return blob


def ResNetConvXBody(in_blob, block_counts, on_stage_end=lambda x: x):
    dl_net = in_blob.dl_net()
    assert (len(block_counts) == 3 or len(block_counts) == 4)
    filters = [256, 512, 1024, 2048]
    filters_inner = [64, 128, 256, 512]
    blob = in_blob
    for i in range(len(block_counts)):
        blob = ResidualStage(blob, "res%d" % (i + 2), block_counts[i],
                             filters[i], filters_inner[i], 1 if i == 0 else 2)

        on_stage_end(blob)

        prob_list_key.append("res%d" % (i+2))
        prob_list.append(blob)

    return blob


def ImgClassifyDecoder(dlnet, data_dir=''):
    return dlnet.DecodeOFRecord(data_dir, name='decode', blob=[
        {
            'name': 'encoded',
            'shape': {'dim': [IMAGE_SIZE, IMAGE_SIZE, 3]},
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


prob_list_key = []
prob_list = []


def BuildWithDeprecatedAPI(data_dir):
    dl_net = flow.deprecated.get_cur_job_dlnet_builder()

    decoders = ImgClassifyDecoder(
        dl_net, data_dir=data_dir)
    img_blob = decoders['encoded']
    transposers = dl_net.Transpose(
        img_blob, name='transpose', perm=[0, 3, 1, 2])

    # input blobs
    label_blob = decoders['class/label']
    img_blob = transposers

    prob_list_key.append("input_img")
    prob_list.append(img_blob)

    with dl_net.VariableScope('Resnet'):
        block_counts = [3, 4, 6, 3]
        blob = ResNetConvXBody(BasicStem(img_blob), block_counts, lambda x: x)
        pool5 = dl_net.AveragePooling2D(blob, name="pool5", pool_size=[7, 7], strides=[
                                        1, 1], padding="valid", data_format='channels_first')

        prob_list_key.append("pool5")
        prob_list.append(pool5)

        blob = Reshape(pool5, [-1, 2048])
        #blob = Reshape(blob, [BATCH_SIZE, -1])
        # TODO: Reshape result is not right
        # TODO: how to get 2048??

        # blob = _FullyConnected(blob, name="fc1001",
        #                       input_size=2048, units=1001, **_BASIC_FC_CONF)

        fc1001 = dl_net.FullyConnected(
            blob, name="fc1001", units=1001, **_BASIC_FC_CONF)
        prob_list_key.append("fc1001")
        prob_list.append(fc1001)

        softmax = dl_net.Softmax(fc1001, name='softmax')
        cross_entropy = dl_net.SparseCrossEntropy(
            softmax, label_blob, name='cross_entropy')

    prob_list_key.append("cross_entropy")
    prob_list.append(cross_entropy)
    return prob_list


# -------------------------------------------------------------------- #
# Oneflow config
# -------------------------------------------------------------------- #

_MODEL_SAVE = "./output/model_save-{}".format(
    str(datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))

_MODEL_LOAD = "/home/qiaojing/dev/cnn/cnns_test/dev_job_set_branch_test/resnet50/models/of_model"

_DATA_DIR = "/dataset/imagenet_224/train/32"
_SINGLE_PIC_DATA_DIR = '/dataset/PNGS/PNG228/of_record'
_REPEATED_PIC_DATA_DIR = '/dataset/PNGS/PNG228/of_record_repeated'
_EVAL_DIR = _REPEATED_PIC_DATA_DIR
_TRAIN_DIR = _REPEATED_PIC_DATA_DIR
DATA_PART_NUM = 32  # 32 or 1
BATCH_SIZE = 4
NUM_ITER = 5


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_num_per_node',
                        type=int, default=1, required=False)
    parser.add_argument('-m', '--multinode', default=False,
                        action="store_true", required=False)
    parser.add_argument('-s', '--skip_scp_binary', default=False,
                        action="store_true", required=False)
    parser.add_argument('-c', '--scp_binary_without_uuid', default=False,
                        action="store_true", required=False)

    return parser.parse_args(argv)


def TrainAlexNet():
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(BATCH_SIZE).data_part_num(
        DATA_PART_NUM).default_data_type(flow.float)
    job_conf.train_conf()
    job_conf.train_conf().primary_lr = 0.0032
    job_conf.train_conf().num_of_batches_in_snapshot = 100
    job_conf.train_conf().model_update_conf.naive_conf.SetInParent()
    job_conf.train_conf().loss_lbn.extend(["Resnet-cross_entropy/out"])
    return BuildWithDeprecatedAPI(_TRAIN_DIR)


def EvaluateAlexNet():
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(BATCH_SIZE).data_part_num(
        DATA_PART_NUM).default_data_type(flow.float)
    return BuildWithDeprecatedAPI(_EVAL_DIR)


def main(args):
    config = flow.ConfigProtoBuilder()
    config.gpu_device_num(args.gpu_num_per_node)
    config.grpc_use_no_signal()
    #config.model_load_snapshot_path(_MODEL_LOAD)
    config.model_save_snapshots_path(_MODEL_SAVE)
    config.log_dir("./output/log")
    config.ctrl_port(12138)

    if args.multinode:
        config.ctrl_port(12139)
        config.machine([{'addr': '192.168.1.12'}, {'addr': '192.168.1.14'}])
        if args.scp_binary_without_uuid:
            flow.deprecated.init_worker(
                config, scp_binary=True, use_uuid=False)
        elif args.skip_scp_binary:
            flow.deprecated.init_worker(
                config, scp_binary=False, use_uuid=False)
        else:
            flow.deprecated.init_worker(config, scp_binary=True, use_uuid=True)

    flow.init(config)
    flow.add_job(TrainAlexNet)
    # flow.add_job(EvaluateAlexNet)

    with flow.Session() as sess:
        check_point = flow.train.CheckPoint()
        check_point.restore().initialize_or_restore(session=sess)

        fmt_str = '{:>12}  {:>12}  {}'
        print('{:>12}  {:>12}  {:>12}'.format(
            "iter", "loss type", "loss value"))
        for i in range(NUM_ITER):
            prob_output = sess.run(TrainAlexNet).get()
            prob_dict = dict(zip(prob_list_key, prob_output))

            print(fmt_str.format(i, "train loss:",
                                 prob_dict["cross_entropy"]))

            hook.dump_tensor_to_file(
                prob_dict, "./prob_output/iter_{}".format(i))

#            if (i + 1) % 100 is 0:
#                print(fmt_str.format(i, "eval loss:", sess.run(
#                    EvaluateAlexNet).get().mean()))
#            if (i + 1) % 100 is 0:
#                check_point.save(session=sess)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
