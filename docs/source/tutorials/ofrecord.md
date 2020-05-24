# 制作OFRecord数据集

OneFlow 支持传入的数据格式为 OFRecord，OFRecord 是包含原始图片信息的二进制文件。直接使用二进制文件训练，不再一张一张地读取离散图片，可以减少图片读取时的数据阻塞，大大提高训练效率。

如果想使用真实数据训练，用户需要提前把图片预处理成 OneFlow 可以训练的 OFRecord。

本示例将介绍如何利用脚本预处理 ImageNet 数据集，生成 OFRecord 数据。

## 1. 下载源文件

将所需源文件下载到本地目录下。
>$ git clone https://github.com/Oneflow-Inc/OFRecord.git 

将 ImageNet 数据集转化成 OFRecord 的脚本及文件存放在 /OFRecord/gen_imagenet_ofrecord/ 路径下。

## 2. 脚本使用

实际上，OneFlow 中使用的数据集是指经过了预处理，分成了不同 `part-0~[n-1]` 的 `n` （`0~n-1`）个批次的 OFRecord 二进制图片数据文件。
OneFlow 提供了一个一键执行的脚本 preprocess_raw_pics.py，可以直接将 ImageNet 的原始图集预处理为待训练的 OFRecord 数据集。

本示例在还未下载源文件数据包的前提下，期待生成一个有 60 个 `part` 的 `validation` 图集，并在图片分配时利用 60 个线程去处理，则在终端中输入如下命令：
>$ python gen_imagenet_ofrecord.py --name validation --train_shards 60 --threads_num 60 --data_file_type validation --dest_dir .

但由于 ImageNet 数据集太大，脚本下载时间过长，所以，通常更建议先利用网络下载 tar 包，直接进行之后的解压和制作处理。

去官网下载好完整的 tar 包，即 annotations.tar.gz，ILSVRC2012_img_train.tar，ILSVRC2012_img_val.tar。
下载完成后，直接进行后续处理，在终端中输入如下命令：
则在终端中输入如下命令：
>$ python gen_imagenet_ofrecord.py --train_shards 60 --threads_num 60 --data_file_type validation --dest_dir . -id --name validation


参数含义：
- `--is_downloaded`：是否已经下载 MNIST 数据包。若不添加参数，说明用户没有下载，需要下载；若添加参数，说明用户已经下载，不再重复下载。
- `--train_shards`：将目标图集分成多少个图集的图集批次数。
- `--num_threads`：由于图片数量巨大，耗时较长，通常会采用多线程处理，该参数为处理的线程数。
- `--data_file`：原始图集分为 train 和 validation 两种，该参数用于指定处理训练图集（`train`）还是处理验证图集（`validation`）。
设置时，`train_shards` 可随意设置，但始终应该是 `num_threads` 的整数倍。


运行中提示输入：

```bash
In order to download the imagenet data, you have to create an account with
image-net.org. This will get you a username and an access key. You can set the
IMAGENET_USERNAME and IMAGENET_ACCESS_KEY environment variables, or you can
enter the credentials here.

Username:
Access key:
```
在命令行中即时输入任意用户名和密码即可。

例如，当输入参数 `validation` 时，经过较长时间的处理，打印显示：
```bash 
 Finished writing all 50000 images in data set.
```
则图集处理成功。

 运行成功，即可在 /OFRecord/ 目录下生成的 imagenet_60/train 或者 imagenet_60/val (或者两者皆有) 路径下找到完成预处理的图集 `part-0` ~ `part-59`。