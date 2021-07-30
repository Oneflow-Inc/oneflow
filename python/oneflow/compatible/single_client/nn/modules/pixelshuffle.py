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
from oneflow.compatible.single_client.framework.tensor import Tensor
from oneflow.compatible.single_client.nn.module import Module


class PixelShuffle(Module):
    """The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle

    Rearranges elements in a tensor of shape :math:`(*, C \\times r^2, H, W)`
    to a tensor of shape :math:`(*, C, H \\times r, W \\times r)`, where r is an upscale factor.

    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.

    See the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details.

    Args:
        upscale_factor (int): factor to increase spatial resolution by

    Shape:
        - Input: :math:`(*, C_{in}, H_{in}, W_{in})`, where * is zero or more batch dimensions
        - Output: :math:`(*, C_{out}, H_{out}, W_{out})`, where

    .. math::
        C_{out} = C_{in} \\div \\text{upscale_factor}^2

    .. math::
        H_{out} = H_{in} \\times \\text{upscale_factor}

    .. math::
        W_{out} = W_{in} \\times \\text{upscale_factor}

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> m = flow.nn.PixelShuffle(upscale_factor=2)
        >>> x = flow.Tensor(np.random.randn(3, 4, 5, 5))
        >>> y = m(x)
        >>> print(y.size())
        flow.Size([3, 1, 10, 10])

        >>> m = flow.nn.PixelShuffle(upscale_factor=3)
        >>> x = flow.Tensor(np.random.randn(1, 18, 2, 2))
        >>> y = m(x)
        >>> print(y.size())
        flow.Size([1, 2, 6, 6])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        assert upscale_factor > 0, "The scale factor must larger than zero"
        self.upscale_factor = upscale_factor

    def forward(self, input: Tensor) -> Tensor:
        assert len(input.shape) == 4, "Only Accept 4D Tensor"
        (_batch, _channel, _height, _width) = input.shape
        assert (
            _channel % self.upscale_factor ** 2 == 0
        ), "The channels of input tensor must be divisible by (upscale_factor * upscale_factor)"
        _new_c = int(_channel / self.upscale_factor ** 2)
        out = input.reshape([_batch, _new_c, self.upscale_factor ** 2, _height, _width])
        out = out.reshape(
            [_batch, _new_c, self.upscale_factor, self.upscale_factor, _height, _width]
        )
        out = out.permute(0, 1, 4, 2, 5, 3)
        out = out.reshape(
            [
                _batch,
                _new_c,
                _height * self.upscale_factor,
                _width * self.upscale_factor,
            ]
        )
        return out


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
