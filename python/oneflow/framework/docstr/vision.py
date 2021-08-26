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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow.F.pad,
    r"""
    pad(input: Tensor, pad: List[int], mode: str = "constant", value: Scalar = 0) -> Tensor

    Args:
        input (Tensor): N-dimensional tensor
        pad (List[int]): 4-elements List
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``        
        value: fill value for ``'constant'`` padding. Default: ``0``

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> pad = [2, 2, 1, 1]
        >>> input = flow.Tensor(np.arange(18).reshape((1, 2, 3, 3)).astype(np.float32))
        >>> output = flow.F.pad(input, pad, mode = "replicate")
        >>> output.shape
        flow.Size([1, 2, 5, 7])
        >>> output
        tensor([[[[ 0.,  0.,  0.,  1.,  2.,  2.,  2.],
                  [ 0.,  0.,  0.,  1.,  2.,  2.,  2.],
                  [ 3.,  3.,  3.,  4.,  5.,  5.,  5.],
                  [ 6.,  6.,  6.,  7.,  8.,  8.,  8.],
                  [ 6.,  6.,  6.,  7.,  8.,  8.,  8.]],
        <BLANKLINE>
                 [[ 9.,  9.,  9., 10., 11., 11., 11.],
                  [ 9.,  9.,  9., 10., 11., 11., 11.],
                  [12., 12., 12., 13., 14., 14., 14.],
                  [15., 15., 15., 16., 17., 17., 17.],
                  [15., 15., 15., 16., 17., 17., 17.]]]], dtype=oneflow.float32)

    See :class:`oneflow.nn.ConstantPad2d`, :class:`oneflow.nn.ReflectionPad2d`, and :class:`oneflow.nn.ReplicationPad2d` for concrete examples on how each of the padding modes works.
        
    """,
)
add_docstr(
    oneflow.F.upsample,
    r"""
    upsample(x: Tensor, height_scale: Float, width_scale: Float, align_corners: Bool, interpolation: str, data_format: str = "channels_first") -> Tensor
  
    Upsample a given multi-channel 2D (spatial) data.

    The input data is assumed to be of the form
    `minibatch x channels x height x width`.
    Hence, for spatial inputs, we expect a 4D Tensor.

    The algorithms available for upsampling are nearest neighbor,
    bilinear, 4D input Tensor, respectively.

    Args:
        height_scale (float):
            multiplier for spatial size. Has to match input size if it is a tuple.
        
        width_scale (float):
            multiplier for spatial size. Has to match input size if it is a tuple.

        align_corners (bool): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is ``'bilinear'``.            

        interpolation (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'bilinear'``.        

        data_format (str, optional): Default: ``'channels_first'``

    Shape:
        - Input: : :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` , where
  
    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{height_scale} \right\rfloor

    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{width_scale} \right\rfloor

  
    For example:

    .. code-block:: python

        >>> import numpy as np

        >>> import oneflow as flow

        >>> input = flow.Tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)  
        >>> output = flow.F.upsample(input, height_scale=2.0, width_scale=2.0, align_corners=False, interpolation="nearest")
    
        >>> output
        tensor([[[[1., 1., 2., 2.],
                  [1., 1., 2., 2.],
                  [3., 3., 4., 4.],
                  [3., 3., 4., 4.]]]], dtype=oneflow.float32)

    See :class:`~oneflow.nn.Upsample` for more details.

    """,
)
