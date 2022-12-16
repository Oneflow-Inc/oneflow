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
from oneflow.nn.modules.module import Module


class Fold(Module):
    r"""
    Fold(output_size, kernel_size, dilation=1, padding=0, stride=1)

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.Fold.html.

    Combines an array of sliding local blocks into a large containing
    tensor, it also called `col2img`

    Consider a batched :attr:`input` tensor containing sliding local blocks,
    e.g., patches of images, of shape :math:`(N, C \times  \prod(\text{kernel_size}), L)`,
    where :math:`N` is batch dimension, :math:`C \times \prod(\text{kernel_size})`
    is the number of values within a block (a block has :math:`\prod(\text{kernel_size})`
    spatial locations each containing a :math:`C`-channeled vector), and
    :math:`L` is the total number of blocks. (This is exactly the
    same specification as the output shape of :class:`~oneflow.nn.Unfold`.) This
    operation combines these local blocks into the large :attr:`output` tensor
    of shape :math:`(N, C, \text{output_size}[0], \text{output_size}[1], \dots)`
    by summing the overlapping values. Similar to :class:`~oneflow.nn.Unfold`, the
    arguments must satisfy

    .. math::
        L = \prod_d \left\lfloor\frac{\text{output_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    where :math:`d` is over all spatial dimensions.

    * :attr:`output_size` describes the spatial shape of the large containing
      tensor of the sliding local blocks. It is useful to resolve the ambiguity
      when multiple input shapes map to same number of sliding blocks, e.g.,
      with ``stride > 0``.

    The :attr:`padding`, :attr:`stride` and :attr:`dilation` arguments specify
    how the sliding blocks are retrieved.

    * :attr:`stride` controls the stride for the sliding blocks.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension before
      reshaping.

    * :attr:`dilation` controls the spacing between the kernel points; also known as 
      the à trous algorithm.

    Args:
        output_size (int or tuple): the shape of the spatial dimensions of the
                                    output (i.e., ``output.sizes()[2:]``)
        kernel_size (int or tuple): the size of the sliding blocks
        stride (int or tuple): the stride of the sliding blocks in the input
                               spatial dimensions. Default: 1
        padding (int or tuple, optional): implicit zero padding to be added on
                                          both sides of input. Default: 0
        dilation (int or tuple, optional): a parameter that controls the
                                           stride of elements within the
                                           neighborhood. Default: 1

    * If :attr:`output_size`, :attr:`kernel_size`, :attr:`dilation`,
      :attr:`padding` or :attr:`stride` is an int or a tuple of length 1 then
      their values will be replicated across all spatial dimensions.

    * For the case of two output spatial dimensions this operation is sometimes
      called ``col2im``.

    .. note::
        :class:`~oneflow.nn.Fold` calculates each combined value in the resulting
        large tensor by summing all values from all containing blocks.
        :class:`~oneflow.nn.Unfold` extracts the values in the local blocks by
        copying from the large tensor. So, if the blocks overlap, they are not
        inverses of each other.

        In general, folding and unfolding operations are related as
        follows. Consider :class:`~oneflow.nn.Fold` and
        :class:`~oneflow.nn.Unfold` instances created with the same
        parameters:

        >>> fold_params = dict(kernel_size=..., dilation=..., padding=..., stride=...)
        >>> fold = nn.Fold(output_size=..., **fold_params)
        >>> unfold = nn.Unfold(**fold_params)

        Then for any (supported) ``input`` tensor the following
        equality holds:

        ::

            fold(unfold(input)) == divisor * input

        where ``divisor`` is a tensor that depends only on the shape
        and dtype of the ``input``:

        >>> input_ones = oneflow.ones(input.shape, dtype=input.dtype)
        >>> divisor = fold(unfold(input_ones))

        When the ``divisor`` tensor contains no zero elements, then
        ``fold`` and ``unfold`` operations are inverses of each
        other (up to constant divisor).

    .. warning::
        Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.

    Shape:
        - Input: :math:`(N, C \times \prod(\text{kernel_size}), L)` or :math:`(C \times \prod(\text{kernel_size}), L)`
        - Output: :math:`(N, C, \text{output_size}[0], \text{output_size}[1], \dots)`
          or :math:`(C, \text{output_size}[0], \text{output_size}[1], \dots)` as described above

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
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        return flow._C.fold(
            input,
            self.output_size,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
            "channels_first",
        )

    def extra_repr(self) -> str:
        return (
            "output_size={output_size}, kernel_size={kernel_size}, "
            "dilation={dilation}, padding={padding}, stride={stride}".format(
                **self.__dict__
            )
        )


class Unfold(Module):
    r"""
    Unfold(kernel_size, dilation=1, padding=0, stride=1)

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.Unfold.html.

    This op extracts elements in a local window from input tensor, it also called `img2col`. 

    Consider a batched :attr:`input` tensor of shape :math:`(N, C, *)`,
    where :math:`N` is the batch dimension, :math:`C` is the channel dimension,
    and :math:`*` represent arbitrary spatial dimensions. This operation flattens
    each sliding :attr:`kernel_size`-sized block within the spatial dimensions
    of :attr:`input` into a column (i.e., last dimension) of a 3-D :attr:`output`
    tensor of shape :math:`(N, C \times \prod(\text{kernel_size}), L)`, where
    :math:`C \times \prod(\text{kernel_size})` is the total number of values
    within each block (a block has :math:`\prod(\text{kernel_size})` spatial
    locations each containing a :math:`C`-channeled vector), and :math:`L` is
    the total number of such blocks:

    .. math::
        L = \prod_d \left\lfloor\frac{\text{spatial_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    where :math:`\text{spatial_size}` is formed by the spatial dimensions
    of :attr:`input` (:math:`*` above), and :math:`d` is over all spatial
    dimensions.

    Therefore, indexing :attr:`output` at the last dimension (column dimension)
    gives all values within a certain block.

    The :attr:`padding`, :attr:`stride` and :attr:`dilation` arguments specify
    how the sliding blocks are retrieved.

    * :attr:`stride` controls the stride for the sliding blocks.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension before
      reshaping.

    * :attr:`dilation` controls the spacing between the kernel points; also known as
      the à trous algorithm.

    Args:
        kernel_size (int or tuple): the size of the sliding blocks
        stride (int or tuple, optional): the stride of the sliding blocks in the input
                                         spatial dimensions. Default: 1
        padding (int or tuple, optional): implicit zero padding to be added on
                                          both sides of input. Default: 0
        dilation (int or tuple, optional): a parameter that controls the
                                           stride of elements within the
                                           neighborhood. Default: 1

    * If :attr:`kernel_size`, :attr:`dilation`, :attr:`padding` or
      :attr:`stride` is an int or a tuple of length 1, their values will be
      replicated across all spatial dimensions.

    * For the case of two input spatial dimensions this operation is sometimes
      called ``im2col``.

    .. note::
        :class:`~oneflow.nn.Fold` calculates each combined value in the resulting
        large tensor by summing all values from all containing blocks.
        :class:`~oneflow.nn.Unfold` extracts the values in the local blocks by
        copying from the large tensor. So, if the blocks overlap, they are not
        inverses of each other.

        In general, folding and unfolding operations are related as
        follows. Consider :class:`~oneflow.nn.Fold` and
        :class:`~oneflow.nn.Unfold` instances created with the same
        parameters:

        >>> fold_params = dict(kernel_size=..., dilation=..., padding=..., stride=...)
        >>> fold = nn.Fold(output_size=..., **fold_params)
        >>> unfold = nn.Unfold(**fold_params)

        Then for any (supported) ``input`` tensor the following
        equality holds:

        ::
                    fold(unfold(input)) == divisor * input

        where ``divisor`` is a tensor that depends only on the shape
        and dtype of the ``input``:

        >>> input_ones = oneflow.ones(input.shape, dtype=input.dtype)
        >>> divisor = fold(unfold(input_ones))

        When the ``divisor`` tensor contains no zero elements, then
        ``fold`` and ``unfold`` operations are inverses of each
        other (up to constant divisor).

    .. warning::
        Currently, only 4-D input tensors (batched image-like tensors) are
        supported.

    Shape:
        - Input: :math:`(N, C, *)`
        - Output: :math:`(N, C \times \prod(\text{kernel_size}), L)` as described above

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
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        return flow._C.unfold(
            input,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
            "channels_first",
        )

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, dilation={dilation}, padding={padding},"
            " stride={stride}".format(**self.__dict__)
        )
