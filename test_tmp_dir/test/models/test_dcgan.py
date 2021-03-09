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
import oneflow.typing as oft
import numpy as np
import os
import unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
def test_1n1c(test_case):
    dcgan = DCGAN()
    dcgan.compare_with_tf(1)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
def test_1n4c(test_case):
    dcgan = DCGAN()
    dcgan.compare_with_tf(4)


class DCGAN:
    def __init__(self):
        self.lr = 1e-4
        self.z_dim = 100
        self.batch_size = 32

    def compare_with_tf(self, gpu_num, result_dir="/dataset/gan_test/dcgan/"):
        flow.config.gpu_device_num(gpu_num)
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(type="train", function_config=func_config)
        def test_generator(
            z: oft.Numpy.Placeholder((self.batch_size, self.z_dim)),
            label1: oft.Numpy.Placeholder((self.batch_size, 1)),
        ):
            g_out = self.generator(z, trainable=True, const_init=True)
            g_logits = self.discriminator(g_out, trainable=False, const_init=True)
            g_loss = flow.nn.sigmoid_cross_entropy_with_logits(
                flow.ones_like(g_logits),
                g_logits,
                name="Gloss_sigmoid_cross_entropy_with_logits",
            )

            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [self.lr]), momentum=0
            ).minimize(g_loss)
            return g_loss

        @flow.global_function(type="train", function_config=func_config)
        def test_discriminator(
            z: oft.Numpy.Placeholder((self.batch_size, 100)),
            images: oft.Numpy.Placeholder((self.batch_size, 1, 28, 28)),
            label1: oft.Numpy.Placeholder((self.batch_size, 1)),
            label0: oft.Numpy.Placeholder((self.batch_size, 1)),
        ):
            g_out = self.generator(z, trainable=False, const_init=True)
            g_logits = self.discriminator(g_out, trainable=True, const_init=True)
            d_loss_fake = flow.nn.sigmoid_cross_entropy_with_logits(
                flow.zeros_like(g_logits),
                g_logits,
                name="Dloss_fake_sigmoid_cross_entropy_with_logits",
            )

            d_logits = self.discriminator(
                images, trainable=True, reuse=True, const_init=True
            )
            d_loss_real = flow.nn.sigmoid_cross_entropy_with_logits(
                flow.ones_like(d_logits),
                d_logits,
                name="Dloss_real_sigmoid_cross_entropy_with_logits",
            )
            d_loss = d_loss_fake + d_loss_real
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [self.lr]), momentum=0
            ).minimize(d_loss)

            return d_loss

        check_point = flow.train.CheckPoint()
        check_point.init()

        z = np.load(os.path.join(result_dir, "z.npy"))
        imgs = np.load(os.path.join(result_dir, "img.npy")).transpose(0, 3, 1, 2)
        label1 = np.ones((self.batch_size, 1)).astype(np.float32)
        label0 = np.zeros((self.batch_size, 1)).astype(np.float32)
        g_loss = test_generator(z, label1).get()
        d_loss = test_discriminator(z, imgs, label1, label0).get()
        tf_g_loss = np.load(os.path.join(result_dir, "g_loss.npy"))
        tf_d_loss = np.load(os.path.join(result_dir, "d_loss.npy"))

        if gpu_num == 1:  # multi-gpu result can not pass
            assert np.allclose(
                g_loss.numpy(), tf_g_loss, rtol=1e-2, atol=1e-1
            ), "{}-{}".format(g_loss.ndarray().mean(), tf_g_loss.mean())
            assert np.allclose(
                d_loss.numpy(), tf_d_loss, rtol=1e-2, atol=1e-1
            ), "{}-{}".format(d_loss.ndarray().mean(), tf_d_loss.mean())

    def generator(self, z, const_init=False, trainable=True):
        # (n, 256, 7, 7)
        h0 = layers.dense(
            z, 7 * 7 * 256, name="g_fc1", const_init=const_init, trainable=trainable
        )
        h0 = layers.batchnorm(h0, axis=1, name="g_bn1")
        h0 = flow.nn.leaky_relu(h0, 0.3)
        h0 = flow.reshape(h0, (-1, 256, 7, 7))
        # (n, 128, 7, 7)
        h1 = layers.deconv2d(
            h0,
            128,
            5,
            strides=1,
            name="g_deconv1",
            const_init=const_init,
            trainable=trainable,
        )
        h1 = layers.batchnorm(h1, name="g_bn2")
        h1 = flow.nn.leaky_relu(h1, 0.3)
        # (n, 64, 14, 14)
        h2 = layers.deconv2d(
            h1,
            64,
            5,
            strides=2,
            name="g_deconv2",
            const_init=const_init,
            trainable=trainable,
        )
        h2 = layers.batchnorm(h2, name="g_bn3")
        h2 = flow.nn.leaky_relu(h2, 0.3)
        # (n, 1, 28, 28)
        out = layers.deconv2d(
            h2,
            1,
            5,
            strides=2,
            name="g_deconv3",
            const_init=const_init,
            trainable=trainable,
        )
        out = flow.math.tanh(out)
        return out

    def discriminator(self, img, const_init=False, trainable=True, reuse=False):
        # (n, 1, 28, 28)
        h0 = layers.conv2d(
            img,
            64,
            5,
            name="d_conv1",
            const_init=const_init,
            trainable=trainable,
            reuse=reuse,
        )
        h0 = flow.nn.leaky_relu(h0, 0.3)
        # h0 = flow.nn.dropout(h0, rate=0.3)
        # (n, 64, 14, 14)
        h1 = layers.conv2d(
            h0,
            128,
            5,
            name="d_conv2",
            const_init=const_init,
            trainable=trainable,
            reuse=reuse,
        )
        h1 = flow.nn.leaky_relu(h1, 0.3)
        # h1 = flow.nn.dropout(h1, rate=0.3)
        # (n, 128 * 7 * 7)
        out = flow.reshape(h1, (self.batch_size, -1))
        # (n, 1)
        out = layers.dense(
            out, 1, name="d_fc", const_init=const_init, trainable=trainable, reuse=reuse
        )
        return out


class layers:
    @classmethod
    def deconv2d(
        cls,
        input,
        filters,
        size,
        name,
        strides=2,
        trainable=True,
        reuse=False,
        const_init=False,
        use_bias=False,
    ):
        name_ = name if reuse == False else name + "_reuse"
        # weight : [in_channels, out_channels, height, width]
        weight_shape = (input.shape[1], filters, size, size)
        output_shape = (
            input.shape[0],
            filters,
            input.shape[2] * strides,
            input.shape[3] * strides,
        )

        weight = flow.get_variable(
            name + "-weight",
            shape=weight_shape,
            dtype=input.dtype,
            initializer=flow.random_normal_initializer(stddev=0.02)
            if not const_init
            else flow.constant_initializer(0.002),
            trainable=trainable,
            reuse=reuse,
        )

        output = flow.nn.conv2d_transpose(
            input,
            weight,
            strides=[strides, strides],
            output_shape=output_shape,
            padding="SAME",
            data_format="NCHW",
            name=name_,
        )

        if use_bias:
            bias = flow.get_variable(
                name + "-bias",
                shape=(filters,),
                dtype=input.dtype,
                initializer=flow.constant_initializer(0.0),
                trainable=trainable,
                reuse=reuse,
            )

            output = flow.nn.bias_add(output, bias, "NCHW")
        return output

    @classmethod
    def conv2d(
        cls,
        input,
        filters,
        size,
        name,
        strides=2,
        padding="same",
        trainable=True,
        reuse=False,
        const_init=False,
        use_bias=True,
    ):
        name_ = name if reuse == False else name + "_reuse"

        # (output_dim, k_h, k_w, input.shape[3]) if NHWC
        weight_shape = (filters, input.shape[1], size, size)
        weight = flow.get_variable(
            name + "-weight",
            shape=weight_shape,
            dtype=input.dtype,
            initializer=flow.random_normal_initializer(stddev=0.02)
            if not const_init
            else flow.constant_initializer(0.002),
            trainable=trainable,
            reuse=reuse,
        )

        output = flow.nn.compat_conv2d(
            input,
            weight,
            strides=[strides, strides],
            padding=padding,
            data_format="NCHW",
            name=name_,
        )

        if use_bias:
            bias = flow.get_variable(
                name + "-bias",
                shape=(filters,),
                dtype=input.dtype,
                initializer=flow.constant_initializer(0.0),
                trainable=trainable,
                reuse=reuse,
            )

            output = flow.nn.bias_add(output, bias, "NCHW")
        return output

    @classmethod
    def dense(
        cls,
        input,
        units,
        name,
        use_bias=False,
        trainable=True,
        reuse=False,
        const_init=False,
    ):
        name_ = name if reuse == False else name + "_reuse"

        in_shape = input.shape
        in_num_axes = len(in_shape)
        assert in_num_axes >= 2

        inputs = flow.reshape(input, (-1, in_shape[-1])) if in_num_axes > 2 else input

        weight = flow.get_variable(
            name="{}-weight".format(name),
            shape=(units, inputs.shape[1]),
            dtype=inputs.dtype,
            initializer=flow.random_normal_initializer(stddev=0.02)
            if not const_init
            else flow.constant_initializer(0.002),
            trainable=trainable,
            model_name="weight",
            reuse=reuse,
        )

        out = flow.matmul(a=inputs, b=weight, transpose_b=True, name=name_ + "matmul",)

        if use_bias:
            bias = flow.get_variable(
                name="{}-bias".format(name),
                shape=(units,),
                dtype=inputs.dtype,
                initializer=flow.random_normal_initializer()
                if not const_init
                else flow.constant_initializer(0.002),
                trainable=trainable,
                model_name="bias",
                reuse=reuse,
            )
            out = flow.nn.bias_add(out, bias, name=name_ + "_bias_add")

        out = flow.reshape(out, in_shape[:-1] + (units,)) if in_num_axes > 2 else out
        return out

    @classmethod
    def batchnorm(cls, input, name, axis=1, reuse=False):
        name_ = name if reuse == False else name + "_reuse"
        return flow.layers.batch_normalization(input, axis=axis, name=name_)
