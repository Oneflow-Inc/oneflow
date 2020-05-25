# AlexNet示例

本文以AlexNet为例，介绍如何使用OneFlow的Python API搭建训练和验证网络，以及如何在单机和多机上运行训练任务和验证任务。

## 数据加载、预处理和解码:

这里，首先使用`flow.data.BlobConf()`接口，构建两个`BlobConf`，分别用于指定从OFRecord格式的数据集中加载图像和标签的方式，OneFlow运行时会根据`BlobConf`将对应的数据加载到对应的`Blob`中。

对于图像数据，在加载时可以指定进行数据预处理的方式，比如可以使用`flow.data.ImageResizePreprocessor(227, 227)`将图片resize成227*227的大小，使用`flow.data.NormByChannelPreprocessor((123.68, 116.78, 103.94))`对图片进行归一化等等。

然后，使用`flow.data.decode_ofrecord()`，对数据进行解码。

```
# 从数据集加载图像，并进行数据预处理
image_blob_conf = flow.data.BlobConf(
    "encoded",
    shape=(227, 227, 3),
    dtype=flow.float,
    codec=flow.data.ImageCodec(
      [flow.data.ImageResizePreprocessor(227, 227)]),
    preprocessors=[
      flow.data.NormByChannelPreprocessor((123.68, 116.78, 103.94))],)

# 从数据集加载标签
label_blob_conf = flow.data.BlobConf(
    "class/label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec())

# 解码
labels, images = flow.data.decode_ofrecord(
    data_dir, (label_blob_conf, image_blob_conf),
    batch_size=12, data_part_num=8, name="decode")
```

## 搭建网络：

使用`flow.nn.conv2d`,`flow.nn.avg_pool2d`,`flow.layers.dense`等算子，搭建AlexNet网络模型。这个过程大多数框架都是类似的，这里就不赘述了。 

```
# 数据数据集格式转换， NHWC -> NCHW
  transposed = flow.transpose(
    images, 
    name="transpose", 
    perm=[0, 3, 1, 2])

# 卷积
conv1 = _conv2d_layer(
    "conv1", 
    transposed, 
    filters=64, 
    kernel_size=11, 
    strides=4, 
    padding="VALID")

# 池化
pool1 = flow.nn.avg_pool2d(
  conv1, 3, 2, 
  "VALID", 
  "NCHW", 
  name="pool1")

# 中间略
...

# 全链接
fc3 = flow.layers.dense(
        inputs=dropout2,
        units=1001,
        activation=None,
        use_bias=False,
        kernel_initializer=flow.random_uniform_initializer(),
        bias_initializer=False,
        trainable=True,
        name="fc3",)

# 损失函数
loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, 
        fc3, 
        name="softmax_loss")
```


## 添加训练任务：

在OneFlow中，通过`@flow.function`装饰器，可以将一个Python函数指定为一个任务(Job)。

这里的任务可以是训练任务，也可以验证或预测任务。

对于训练任务而言，需要特别注意：

（1）需要设置训练超参数，包括学习率的初始值、优化方法等等。它们都是通过`flow.config.train.xxx`接口在训练任务中指定。

（2）需要具体指定在上一步构建的网络模型中，优化目标是什么。一般而言，我们会将网络输出的loss作为优化目标。这里通过`flow.losses.add_loss()`接口指定。

```
# 训练任务
@flow.function
def alexnet_train_job():
    # 设置训练超参数
    flow.config.train.primary_lr(0.00001)
    flow.config.train.model_update_conf(dict(naive_conf={}))

    # 加载数据
    (labels, images) = _data_load_layer(args.train_dir)
    
    # 构建网络
    loss = alexnet(images, labels)

    # 指定训练网络的loss(优化目标)
    flow.losses.add_loss(loss)

    return loss
```


## 添加验证任务：

验证任务和训练任务的区别是，验证任务不会更新网络中的模型参数。

另外，验证网络和训练网络的网络结构可以相同，也可以不同。

这里，对于我们的AlexNet来说，验证网络的网络结构和训练网络是相同的，仅仅是加载的数据集不同，所以直接借用刚才搭建好的网络，构建一个验证任务。

```
# 验证任务
@flow.function
def alexnet_eval_job():
    # 加载数据
    (labels, images) = _data_load_layer(args.eval_dir)

    # 构建验证网络
    loss = alexnet(images, labels)

    return loss
```

## 配置运行方式

这个部分的内容，是OneFlow区别于其它深度学习框架最为独特的部分。

对于OneFlow的用户而言，他们使用OneFlow Python API 写的一个单机的深度学习算法，可以很容易地就扩展成多机分布式程序，在分布式环境下并行执行。

也就是说，用户只需要关心算法本身的逻辑或者深度神经网络的结构，而几乎不用考虑分布式执行的问题，OneFlow框架本身会处理分布式情况下并行数据划分的问题。使得用户只用写好单机单卡的程序，只需要简单的配置，就可以直接以单机多卡，甚至多机多卡的方式运行，并且保证运行效率足够高。

接下来说明需要配置的内容：

（1）通过`flow.config.gpu_device_num()`指定每个节点运行的GPU卡数，这里的节点一般是指机器，即一台机器为一个节点。

（2）通过`flow.config.ctrl_port()`指定控制命令绑定的端口，这里只要指定一个空闲的端口即可。

（3）通过`flow.config.default_data_type()`指定数据类型，可以是`flow.float`类型(全精度)，或者`flow.half`类型(半精度)等。

（4）对于多机运行的情况，还需要通过`flow.config.machine()`指定各个机器的ip地址，并在每个机器运行同样的 python 脚本。

```
# 配置运行方式
flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config.ctrl_port(9788)
flow.config.default_data_type(flow.float)

# 设置多机分布式端口
if args.multinode:
    flow.config.ctrl_port(12138)
    flow.config.machine(
      [{"addr": "192.168.1.15"}, {"addr": "192.168.1.16"}])
```

## 模型加载／初始化

可以通过`flow.train.CheckPoint().init()`接口，直接随机初始化模型。也可以通过`flow.train.CheckPoint().load()`接口，从已有的模型文件，一般我们称为CheckPoint文件，加载到我们即将运行的训练或测试网络中。

```
# alexnet.py
# 模型加载／初始化
check_point = flow.train.CheckPoint()
if not args.model_load_dir:
    check_point.init()
else:
    check_point.load(args.model_load_dir)
```

## 执行训练／验证迭代过程

做好了上面的搭建网络，添加训练／验证任务，配置运行方式等准备工作之后，就可以开始模型的训练迭代过程了。

这里，我们希望每在训练数据集上训练10轮，就在验证数据集上进行一次验证，并分别将训练和验证的loss打印在屏幕上。使用`Job`的`get()`方法，可以拿到之前我们定义的训练任务和验证任务的输出。

```
for i in range(args.iter_num)):
    fmt_str = "{:>12}  {:>12}  {:>12.10f}"
    
    # 打印训练输出
    train_loss = alexnet_train_job().get().mean()
    print(fmt_str.format(i, "train loss:", train_loss))
    
    # 打印验证输出
    if (i + 1) % 10 == 0:
        eval_loss = alexnet_eval_job().get().mean()
        print(fmt_str.format(i, "eval loss:", eval_loss))
```

## 保存模型

最后，可以使用`flow.train.CheckPoint().save()`将训练得到的模型保存到CheckPoint文件中。

```
check_point.save(args.model_save_dir + str(i))
```



## 完整的示例代码
```
# ###################################################################
# alexnet.py
# 使用方法说明：
#     单机运行： python alexnet.py -g 1       
#               -g 指定使用的GPU个数
#     多机运行： python alexnet.py -g 8 -m -n "192.168.1.15,192.168.1.16"
#               -g 指定使用的GPU个数
#               -m 指定使用多机运行
#               -n 指定各个机器ip地址，用逗号分隔
# ###################################################################

import oneflow as flow
import argparse


parser = argparse.ArgumentParser(description="flags for oneflow running")
parser.add_argument(
  "-i", "--iter_num", type=int, default=10, required=False)
parser.add_argument(
  "-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument(
  "-m", "--multinode", default=False, action="store_true", required=False)
parser.add_argument(
  "-n","--node_list", type=str, default=None, required=False)
parser.add_argument(
  "-e", "--eval_dir", type=str, default="./dataset/example", required=False)
parser.add_argument(
  "-t", "--train_dir", type=str, default="./dataset/example", required=False)
parser.add_argument(
  "-load", "--model_load_dir", type=str, default="", required=False)
parser.add_argument(
  "-save", "--model_save_dir", type=str, default="./checkpoints", required=False)

args = parser.parse_args()


def _data_load_layer(data_dir):    
    # 从数据集加载图像，并进行数据预处理
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(227, 227, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec([flow.data.ImageResizePreprocessor(227, 227)]),
        preprocessors=[flow.data.NormByChannelPreprocessor((123.68, 116.78, 103.94))],)
    
    # 从数据集加载标签
    label_blob_conf = flow.data.BlobConf(
        "class/label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec())
    
    # 解码
    labels, images = flow.data.decode_ofrecord(
        data_dir, (label_blob_conf, image_blob_conf),
        batch_size=12, data_part_num=8, name="decode")
    
    return labels, images


def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation="Relu",
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
        if activation == "Relu":
            output = flow.keras.activations.relu(output)
        else:
            raise NotImplementedError

    return output



def alexnet(images, labels):
    # 数据数据集格式转换， NHWC -> NCHW
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

    if len(pool5.shape) > 2:
        pool5 = flow.reshape(pool5, shape=(pool5.static_shape[0], -1))

    fc1 = flow.layers.dense(
        inputs=pool5,
        units=4096,
        activation=flow.keras.activations.relu,
        use_bias=False,
        kernel_initializer=flow.random_uniform_initializer(),
        bias_initializer=False,
        trainable=True,
        name="fc1",)

    dropout1 = flow.nn.dropout(fc1, rate=0.5)

    fc2 = flow.layers.dense(
        inputs=dropout1,
        units=4096,
        activation=flow.keras.activations.relu,
        use_bias=False,
        kernel_initializer=flow.random_uniform_initializer(),
        bias_initializer=False,
        trainable=True,
        name="fc2",)

    dropout2 = flow.nn.dropout(fc2, rate=0.5)

    fc3 = flow.layers.dense(
        inputs=dropout2,
        units=1001,
        activation=None,
        use_bias=False,
        kernel_initializer=flow.random_uniform_initializer(),
        bias_initializer=False,
        trainable=True,
        name="fc3",)

    # 损失函数
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, fc3, name="softmax_loss")

    return loss


# 训练任务
@flow.function
def alexnet_train_job():
    # 设置训练超参数
    flow.config.train.primary_lr(0.00001)
    flow.config.train.model_update_conf(dict(naive_conf={}))

    # 加载数据
    (labels, images) = _data_load_layer(args.train_dir)
    
    # 构建网络
    loss = alexnet(images, labels)

    # 指定训练网络的loss(优化目标)
    flow.losses.add_loss(loss)

    return loss


# 验证任务
@flow.function
def alexnet_eval_job():
    # 加载数据
    (labels, images) = _data_load_layer(args.eval_dir)

    # 构建验证网络
    loss = alexnet(images, labels)

    return loss


def main():
    # 配置运行方式
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.config.ctrl_port(9788)
    flow.config.default_data_type(flow.float)

    # 设置多机分布式端口
    if args.multinode:
        flow.config.ctrl_port(12138)
        nodes = []
        for n in args.node_list.strip().split(","):
            addr_dict = {}
            addr_dict["addr"] = n
            nodes.append(addr_dict)
        flow.config.machine(nodes)

    # 模型加载／初始化
    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)
    
    # 训练迭代过程
    print("{:>12}  {:>12}  {:>12}".format("iter", "loss type", "loss value"))
    for i in range(args.iter_num)):
        fmt_str = "{:>12}  {:>12}  {:>12.10f}"
        
        # 打印训练输出
        train_loss = alexnet_train_job().get().mean()
        print(fmt_str.format(i, "train loss:", train_loss))
        
        # 打印验证输出
        if (i + 1) % 10 == 0:
            eval_loss = alexnet_eval_job().get().mean()
            print(fmt_str.format(i, "eval loss:", eval_loss))
        
        # 保存模型
        if (i + 1) % 100 == 0:
            check_point.save(args.model_save_dir + str(i))


if __name__ == "__main__":
    main()

```
