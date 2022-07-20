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
from oneflow.nn.common_types import _size_2_t
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _pair


class Fold(Module):
    r"""Combines an array of sliding local blocks into a large containing 
    tensor, it also called `col2img`. 

    Consider a batched :attr:`input` tensor containing sliding local blocks,
    e.g., patches of images, of shape :math:`(N, C \times  \prod(\text{kernel\_size}), L)`,
    where :math:`N` is batch dimension, :math:`C \times \prod(\text{kernel\_size})`
    is the number of values within a block (a block has :math:`\prod(\text{kernel\_size})`
    spatial locations each containing a :math:`C`-channeled vector), and
    :math:`L` is the total number of blocks. (This is exactly the
    same specification as the output shape of :class:`~torch.nn.Unfold`.) This
    operation combines these local blocks into the large :attr:`output` tensor
    of shape :math:`(N, C, \text{output\_size}[0], \text{output\_size}[1], \dots)`
    by summing the overlapping values. Similar to :class:`~torch.nn.Unfold`, the
    arguments must satisfy

    .. math::
        L = \prod_d \left\lfloor\frac{\text{output\_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    Args:
        output_size (_size_2_t): The spatial dimension of output tensor. 
        kernel_size (_size_2_t): The size of kernel. 
        dilation (_size_2_t, optional): The dilation rate. Defaults to 1.
        padding (_size_2_t, optional): The padding value. Defaults to 0.
        stride (_size_2_t, optional): The stride of sliding window. Defaults to 1.

    For example: 

    .. code-block:: python 

        >>> import oneflow as flow 
        >>> import numpy as np

        >>> x_tensor = flow.Tensor(np.random.randn(1, 9, 16))
        >>> fold = flow.nn.Fold(output_size=(4, 4), kernel_size=3, padding=1)
        >>> out = fold(x_tensor)
        >>> out.shape
        oneflow.Size([1, 1, 4, 4])

    """

    def __init__(
        self,
        output_size: _size_2_t,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1,
    ) -> None:
        super(Fold, self).__init__()
        self.output_size = output_size
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.stride = _pair(stride)

    def forward(self, input):
        return flow._C.fold(
            input,
            "channels_first",
            self.output_size,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )

    def extra_repr(self) -> str:
        return (
            "output_size={output_size}, kernel_size={kernel_size}, "
            "dilation={dilation}, padding={padding}, stride={stride}".format(
                **self.__dict__
            )
        )


class Unfold(Module):
    r"""This op extracts elements in a local window from input tensor, it also called `img2col`. 

    Consider a batched :attr:`input` tensor of shape :math:`(N, C, *)`,
    where :math:`N` is the batch dimension, :math:`C` is the channel dimension,
    and :math:`*` represent arbitrary spatial dimensions. This operation flattens
    each sliding :attr:`kernel_size`-sized block within the spatial dimensions
    of :attr:`input` into a column (i.e., last dimension) of a 3-D :attr:`output`
    tensor of shape :math:`(N, C \times \prod(\text{kernel\_size}), L)`, where
    :math:`C \times \prod(\text{kernel\_size})` is the total number of values
    within each block (a block has :math:`\prod(\text{kernel\_size})` spatial
    locations each containing a :math:`C`-channeled vector), and :math:`L` is
    the total number of such blocks:

    .. math::
        L = \prod_d \left\lfloor\frac{\text{spatial\_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    where :math:`\text{spatial\_size}` is formed by the spatial dimensions
    of :attr:`input` (:math:`*` above), and :math:`d` is over all spatial
    dimensions.

    Therefore, indexing :attr:`output` at the last dimension (column dimension)
    gives all values within a certain block.


    Args:
        kernel_size (_size_2_t): The size of kernel. 
        dilation (_size_2_t, optional): The dilation rate. Defaults to 1.
        padding (_size_2_t, optional): The padding value. Defaults to 0.
        stride (_size_2_t, optional): The stride of sliding window. Defaults to 1.

    For example: 

    .. code-block:: python 

        >>> import oneflow as flow 
        >>> import numpy as np 

        >>> x_tensor = flow.Tensor(np.random.randn(1, 1, 4, 4))
        >>> unfold = flow.nn.Unfold(kernel_size=3, padding=1)
        >>> out = unfold(x_tensor)
        >>> out.shape
        oneflow.Size([1, 9, 16])

    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1,
    ) -> None:
        super(Unfold, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.stride = _pair(stride)

    def forward(self, input):
        return flow._C.unfold(
            input,
            "channels_first",
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, dilation={dilation}, padding={padding},"
            " stride={stride}".format(**self.__dict__)
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
