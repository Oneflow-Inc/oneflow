"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow


def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation=flow.math.relu,
    use_bias=False,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=flow.random_uniform_initializer(),
):
    weight_shape = (filters, input.shape[1], kernel_size, kernel_size)
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

    if callable(activation):
        output = activation(output)

    return output


def load_data(batch_size, data_dir, data_part_num):
    rgb_mean = [123.68, 116.78, 103.94]
    (image, label) = flow.data.ofrecord_image_classification_reader(
        ofrecord_dir=data_dir,
        batch_size=batch_size,
        data_part_num=data_part_num,
        image_feature_name="encoded",
        label_feature_name="class/label",
        color_space="RGB",
        name="decode",
    )
    res_image = flow.image.resize(image, resize_x=227, resize_y=227, color_space="RGB")
    normal = flow.image.crop_mirror_normalize(
        res_image,
        color_space="RGB",
        output_layout="NCHW",
        mean=rgb_mean,
        output_dtype=flow.float,
    )
    return label, normal


def alexnet(image, label, trainable=True):
    conv1 = _conv2d_layer(
        "conv1", image, filters=64, kernel_size=11, strides=4, padding="VALID",
    )
    pool1 = flow.nn.avg_pool2d(conv1, 3, 2, "VALID", "NCHW", name="pool1")
    conv2 = _conv2d_layer("conv2", pool1, filters=192, kernel_size=5)
    pool2 = flow.nn.avg_pool2d(conv2, 3, 2, "VALID", "NCHW", name="pool2")
    conv3 = _conv2d_layer("conv3", pool2, filters=384)
    conv4 = _conv2d_layer("conv4", conv3, filters=384)
    conv5 = _conv2d_layer("conv5", conv4, filters=256)
    pool5 = flow.nn.avg_pool2d(conv5, 3, 2, "VALID", "NCHW", name="pool5")

    if len(pool5.shape) > 2:
        pool5 = flow.flatten(pool5, start_dim=1, end_dim=-1)

    initializer = flow.truncated_normal_initializer(stddev=0.816496580927726)

    fc1 = flow.layers.dense(
        inputs=pool5,
        units=4096,
        activation=flow.math.relu,
        use_bias=False,
        kernel_initializer=initializer,
        bias_initializer=False,
        trainable=trainable,
        name="fc1",
    )

    dropout1 = fc1
    fc2 = flow.layers.dense(
        inputs=dropout1,
        units=4096,
        activation=flow.math.relu,
        use_bias=False,
        kernel_initializer=initializer,
        bias_initializer=False,
        trainable=trainable,
        name="fc2",
    )

    dropout2 = fc2
    fc3 = flow.layers.dense(
        inputs=dropout2,
        units=1001,
        activation=None,
        use_bias=False,
        kernel_initializer=initializer,
        bias_initializer=False,
        trainable=trainable,
        name="fc3",
    )

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        label, fc3, name="softmax_loss"
    )
    return loss
