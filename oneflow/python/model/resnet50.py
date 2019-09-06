import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from datetime import datetime
import argparse

_DATA_DIR = "/dataset/imagenet_224/train/32"

parser = argparse.ArgumentParser(
    description="flags for multi-node and resource")
parser.add_argument("-n", "--node_num", type=int, default=1)
parser.add_argument("-b", "--batch_size_per_device", type=int, default=8)
parser.add_argument("-g", "--gpu_num_per_node",
                    type=int, default=1, required=False)
parser.add_argument("-t", "--train_dir", type=str,
                    default=_DATA_DIR, required=False)
parser.add_argument("-e", "--eval_dir", type=str,
                    default=_DATA_DIR, required=False)

args = parser.parse_args()


def _data_load_layer(data_dir):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(228, 228, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec([flow.data.ImagePreprocessor("bgr2rgb")]),
        preprocessors=[flow.data.NormByChannelPreprocessor(
            (123.68, 116.78, 103.94))],
    )

    label_blob_conf = flow.data.BlobConf(
        "class/label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        data_dir, (label_blob_conf, image_blob_conf), data_part_num=8, name="decode"
    )


def resnet50(images, labels, trainable=True):
    transposed = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])
    with flow.deprecated.variable_scope("Resnet"):


def TrainNet():
    job_conf = flow.get_cur_job_conf_builder()
    total_batch_size = args.node_num * \
        args.device_num_per_node * args.batch_size_per_device
    job_conf.default_data_type(flow.float).batch_size(total_batch_size)
    job_conf.train_conf()
    job_conf.train_conf().batch_size = total_batch_size
    job_conf.train_conf().primary_lr = 0.0032
    job_conf.train_conf().model_update_conf.naive_conf.SetInParent()

    (labels, images) = _data_load_layer(args.train_dir)
    loss = resnet50(images, labels)
    flow.losses.add_loss(loss)
    return loss
