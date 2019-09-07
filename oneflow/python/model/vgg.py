import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from datetime import datetime
import argparse

_DATA_DIR = "/dataset/PNGS/PNG224/of_record_repeated"
_SINGLE_DATA_DIR = "/dataset/PNGS/PNG224/of_record"
_MODEL_LOAD_DIR = "/dataset/PNGS/cnns_model_for_test/vgg16/models/of_model"
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
parser.add_argument("-e", "--eval_dir", type=str, default=_SINGLE_DATA_DIR, required=False)
parser.add_argument("-t", "--train_dir", type=str, default=_DATA_DIR, required=False)
parser.add_argument("-load", "--model_load_dir", type=str, default=_MODEL_LOAD_DIR, required=False)
#parser.add_argument("-load", "--model_load_dir", type=str, default="", required=False)
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
    padding="VALID",
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=True,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=flow.constant_initializer(),
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
        shape=(224, 224, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec([flow.data.ImagePreprocessor("bgr2rgb")]),
        preprocessors=[flow.data.NormByChannelPreprocessor((123.68, 116.78, 103.94))],
    )

    label_blob_conf = flow.data.BlobConf(
        "class/label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        data_dir, (label_blob_conf, image_blob_conf), data_part_num=32, name="decode"
    )

def _conv_block(in_blob, index, filters, conv_times):
    conv_block = []
    conv_block.insert(0, in_blob)
    for i in range(conv_times):
        conv_i = _conv2d_layer(
        name="conv{}".format(index),
        input=conv_block[i], 
        filters=filters,
        kernel_size=3,
        strides=1,
        )
        conv_block.append(conv_i)
        index += 1
        
    return conv_block
    
    
def vgg(images, labels, trainable=True):
    transposed = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])
    conv1 = _conv_block(transposed, 0, 64, 2)
    print("conv1   ", conv1[-1].shape)  
    pool1 = flow.nn.max_pool2d(conv1[-1], 2, 2, "VALID", "NCHW", name="pool1")
    print("pool1   ", pool1.shape)
    conv2 = _conv_block(pool1, 2, 128, 2)
    
    pool2 = flow.nn.max_pool2d(conv2[-1], 2, 2, "VALID", "NCHW", name="pool2")
    print("pool2    ", pool2.shape)
    conv3 = _conv_block(pool2, 4, 256, 3)

    pool3 = flow.nn.max_pool2d(conv3[-1], 2, 2, "VALID", "NCHW", name="pool3")
    print("pool3   ", pool3.shape)
    conv4 = _conv_block(pool3, 7, 512, 3)

    pool4 = flow.nn.max_pool2d(conv4[-1], 2, 2, "VALID", "NCHW", name="pool4")
    print("pool4    ", pool4.shape)
    conv5 = _conv_block(pool4, 10, 512, 3)

    pool5 = flow.nn.max_pool2d(conv5[-1], 2, 2, "VALID", "NCHW", name="pool5")
    print("pool5   ", pool5.shape)
    def _get_kernel_initializer():
        kernel_initializer = op_conf_util.InitializerConf()
        kernel_initializer.truncated_normal_conf.std = 0.816496580927726
        return kernel_initializer
   
    def _get_bias_initializer():
        bias_initializer = op_conf_util.InitializerConf()
        bias_initializer.constant_conf.value = 0.0
        return bias_initializer

  # if len(pool5.shape) > 2:
    pool5 = flow.reshape(pool5, [-1, 512])
    print("pool5   reshaped  ", pool5.shape)


    fc6 = flow.layers.dense( 
      inputs=pool5, 
      units=4096,
      activation=flow.keras.activations.relu,
      use_bias=True,
      kernel_initializer=_get_kernel_initializer(),
      bias_initializer=_get_bias_initializer(),
      trainable=trainable,
      name="fc1"
    )
    print("fc6   ", fc6.shape)

    fc7 = flow.layers.dense( 
      inputs=fc6, 
      units=4096,
      activation=flow.keras.activations.relu,
      use_bias=True,
      kernel_initializer=_get_kernel_initializer(),
      bias_initializer=_get_bias_initializer(),
      trainable=trainable,
      name="fc2"
    )
    print("fc7    ", fc7.shape)

    fc8 = flow.layers.dense( 
      inputs=fc7, 
      units=1001,
      activation=flow.keras.activations.relu,
      use_bias=True,
      kernel_initializer=_get_kernel_initializer(),
      bias_initializer=_get_bias_initializer(),
      trainable=trainable,
      name="fc_final"
    )
    print("fc8   :", fc8.shape)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, fc8, name="softmax_loss"
    )

    return loss


def TrainNet():
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(8).default_data_type(flow.float)
    job_conf.train_conf()
    job_conf.train_conf().batch_size = 8
    job_conf.train_conf().primary_lr = 0.00001
    job_conf.train_conf().model_update_conf.naive_conf.SetInParent()

    (labels, images) = _data_load_layer(args.train_dir)
    loss = vgg(images, labels)
    flow.losses.add_loss(loss)
    return loss


#def vgg_eval_job():
#    job_conf = flow.get_cur_job_conf_builder()
#    job_conf.batch_size(8).default_data_type(flow.float)
#    (labels, images) = _data_load_layer(args.eval_dir)
#    return vgg(images, labels, False)


if __name__ == "__main__":
    config = flow.ConfigProtoBuilder()
    config.gpu_device_num(args.gpu_num_per_node)
    config.grpc_use_no_signal()
    config.ctrl_port(8888)
    if args.multinode:
        config.ctrl_port(12138)
        config.machine([{"addr": "192.168.1.15"}, {"addr": "192.168.1.16"}])
        if args.remote_by_hand is False:
            if args.scp_binary_without_uuid:
                flow.deprecated.init_worker(config, scp_binary=True, use_uuid=False)
            elif args.skip_scp_binary:
                flow.deprecated.init_worker(config, scp_binary=False, use_uuid=False)
            else:
                flow.deprecated.init_worker(config, scp_binary=True, use_uuid=True)

    flow.init(config)

    flow.add_job(TrainNet)
    #flow.add_job(vgg_eval_job)

    with flow.Session() as sess:
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
                    i, "train loss:", sess.run(TrainNet).get().mean()
                )
            )
          #  if (i + 1) % 10 == 0:
          #      print(
          #          fmt_str.format(
          #              i, "eval loss:", sess.run(vgg_eval_job).get().mean()
          #          )
          #      )
            if (i + 1) % 100 == 0:
                check_point.save(_MODEL_SAVE_DIR + str(i))
        if (
            args.multinode
            and args.skip_scp_binary is False
            and args.scp_binary_without_uuid is False
        ):
            flow.deprecated.delete_worker(config)
