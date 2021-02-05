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

BLOCK_COUNTS = [3, 4, 6, 3]
BLOCK_FILTERS = [256, 512, 1024, 2048]
BLOCK_FILTERS_INNER = [64, 128, 256, 512]


class ResnetBuilder(object):
    def __init__(
        self, weight_regularizer, trainable=True, training=True, channel_last=False
    ):
        self.data_format = "NHWC" if channel_last else "NCHW"
        self.weight_initializer = flow.variance_scaling_initializer(
            2, "fan_in", "random_normal", data_format=self.data_format
        )
        self.weight_regularizer = weight_regularizer
        self.trainable = trainable
        self.training = training

    def _conv2d(
        self, name, input, filters, kernel_size, strides=1, padding="SAME", dilations=1,
    ):
        # There are different shapes of weight metric between 'NCHW' and 'NHWC' mode
        if self.data_format == "NHWC":
            shape = (filters, kernel_size, kernel_size, input.shape[3])
        else:
            shape = (filters, input.shape[1], kernel_size, kernel_size)
        weight = flow.get_variable(
            name + "-weight",
            shape=shape,
            dtype=input.dtype,
            initializer=self.weight_initializer,
            regularizer=self.weight_regularizer,
            model_name="weight",
            trainable=self.trainable,
        )

        return flow.nn.conv2d(
            input, weight, strides, padding, self.data_format, dilations, name=name
        )

    def _batch_norm(self, inputs, name=None, last=False):
        initializer = flow.zeros_initializer() if last else flow.ones_initializer()
        axis = 1
        if self.data_format == "NHWC":
            axis = 3
        return flow.layers.batch_normalization(
            inputs=inputs,
            axis=axis,
            momentum=0.9,  # 97,
            epsilon=1e-5,
            center=True,
            scale=True,
            trainable=self.trainable,
            training=self.training,
            gamma_initializer=initializer,
            moving_variance_initializer=initializer,
            gamma_regularizer=self.weight_regularizer,
            beta_regularizer=self.weight_regularizer,
            name=name,
        )

    def conv2d_affine(
        self, input, name, filters, kernel_size, strides, activation=None, last=False
    ):
        # input data_format must be NCHW, cannot check now
        padding = "SAME" if strides > 1 or kernel_size > 1 else "VALID"
        output = self._conv2d(name, input, filters, kernel_size, strides, padding)
        output = self._batch_norm(output, name + "_bn", last=last)
        if activation == "Relu":
            output = flow.nn.relu(output)

        return output

    def bottleneck_transformation(
        self, input, block_name, filters, filters_inner, strides
    ):
        a = self.conv2d_affine(
            input, block_name + "_branch2a", filters_inner, 1, 1, activation="Relu"
        )

        b = self.conv2d_affine(
            a, block_name + "_branch2b", filters_inner, 3, strides, activation="Relu"
        )

        c = self.conv2d_affine(b, block_name + "_branch2c", filters, 1, 1, last=True)
        return c

    def residual_block(self, input, block_name, filters, filters_inner, strides_init):
        if strides_init != 1 or block_name == "res2_0":
            shortcut = self.conv2d_affine(
                input, block_name + "_branch1", filters, 1, strides_init
            )
        else:
            shortcut = input

        bottleneck = self.bottleneck_transformation(
            input, block_name, filters, filters_inner, strides_init,
        )
        return flow.nn.relu(bottleneck + shortcut)

    def residual_stage(
        self, input, stage_name, counts, filters, filters_inner, stride_init=2
    ):
        output = input
        for i in range(counts):
            block_name = "%s_%d" % (stage_name, i)
            output = self.residual_block(
                output, block_name, filters, filters_inner, stride_init if i == 0 else 1
            )

        return output

    def resnet_conv_x_body(self, input):
        output = input
        for i, (counts, filters, filters_inner) in enumerate(
            zip(BLOCK_COUNTS, BLOCK_FILTERS, BLOCK_FILTERS_INNER)
        ):
            stage_name = "res%d" % (i + 2)
            output = self.residual_stage(
                output, stage_name, counts, filters, filters_inner, 1 if i == 0 else 2
            )

        return output

    def resnet_stem(self, input):
        conv1 = self._conv2d("conv1", input, 64, 7, 2)
        conv1_bn = flow.nn.relu(self._batch_norm(conv1, "conv1_bn"))
        pool1 = flow.nn.max_pool2d(
            conv1_bn,
            ksize=3,
            strides=2,
            padding="SAME",
            data_format=self.data_format,
            name="pool1",
        )
        return pool1


def resnet50(
    images,
    trainable=True,
    need_transpose=False,
    training=True,
    wd=1.0 / 32768,
    channel_last=False,
):
    weight_regularizer = flow.regularizers.l2(wd) if wd > 0.0 and wd < 1.0 else None
    builder = ResnetBuilder(weight_regularizer, trainable, training, channel_last)
    # note: images.shape = (N C H W) in cc's new dataloader, transpose is not needed anymore
    if need_transpose:
        images = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])
    if channel_last:
        # if channel_last=True, then change mode from 'nchw' to 'nhwc'
        images = flow.transpose(images, name="transpose", perm=[0, 2, 3, 1])
    with flow.scope.namespace("Resnet"):
        stem = builder.resnet_stem(images)
        body = builder.resnet_conv_x_body(stem)
        pool5 = flow.nn.avg_pool2d(
            body,
            ksize=7,
            strides=1,
            padding="VALID",
            data_format=builder.data_format,
            name="pool5",
        )
        fc1001 = flow.layers.dense(
            flow.reshape(pool5, (pool5.shape[0], -1)),
            units=1000,
            use_bias=True,
            kernel_initializer=flow.variance_scaling_initializer(
                2, "fan_in", "random_normal"
            ),
            bias_initializer=flow.zeros_initializer(),
            kernel_regularizer=weight_regularizer,
            bias_regularizer=weight_regularizer,
            trainable=trainable,
            name="fc1001",
        )

    return fc1001
