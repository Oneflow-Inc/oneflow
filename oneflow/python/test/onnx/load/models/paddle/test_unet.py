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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from oneflow.python.test.onnx.load.util import load_paddle_module_and_check


class ConvBNReLU(nn.Layer):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding="same", **kwargs
    ):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs
        )

        self._batch_norm = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class UNet(nn.Layer):
    """
    The UNet implementation based on PaddlePaddle.
    The original article refers to
    Olaf Ronneberger, et, al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (https://arxiv.org/abs/1505.04597).
    Args:
        num_classes (int): The unique number of target classes.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.
    """

    def __init__(
        self, num_classes, align_corners=False, use_deconv=False, pretrained=None
    ):
        super().__init__()

        self.encode = Encoder()
        self.decode = Decoder(align_corners, use_deconv=use_deconv)
        self.cls = self.conv = nn.Conv2D(
            in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1
        )

        self.pretrained = pretrained

    def forward(self, x):
        logit_list = []
        x, short_cuts = self.encode(x)
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list


class Encoder(nn.Layer):
    def __init__(self):
        super().__init__()

        self.double_conv = nn.Sequential(ConvBNReLU(3, 64, 3), ConvBNReLU(64, 64, 3))
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]
        self.down_sample_list = nn.LayerList(
            [self.down_sampling(channel[0], channel[1]) for channel in down_channels]
        )

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(ConvBNReLU(in_channels, out_channels, 3))
        modules.append(ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class Decoder(nn.Layer):
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()

        up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.LayerList(
            [
                UpSampling(channel[0], channel[1], align_corners, use_deconv)
                for channel in up_channels
            ]
        )

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x


class UpSampling(nn.Layer):
    def __init__(self, in_channels, out_channels, align_corners, use_deconv=False):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(
                in_channels, out_channels // 2, kernel_size=2, stride=2, padding=0
            )
            in_channels = in_channels + out_channels // 2
        else:
            in_channels *= 2

        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3),
            ConvBNReLU(out_channels, out_channels, 3),
        )

    def forward(self, x, short_cut):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x,
                short_cut.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x


def unet():
    return UNet(num_classes=21)


def test_unet(test_case):
    load_paddle_module_and_check(
        test_case, unet, input_size=(1, 3, 1024, 512), train_flag=False,
    )


from absl import app
from absl.testing import absltest

test_case = absltest.TestCase
test_unet(test_case)
