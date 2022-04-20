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


def grid_sample(
    input,
    grid,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
):
    """The interface is consistent with PyTorch.    
    The documentation is referenced from: 
    https://pytorch.org/docs/1.10/generated/torch.nn.functional.grid_sample.html.

    Given an :attr:`input` and a flow-field :attr:`grid`, computes the
    ``output`` using :attr:`input` values and pixel locations from :attr:`grid`.

    Currently, only spatial (4-D) and volumetric (5-D) :attr:`input` are
    supported.

    In the spatial (4-D) case, for :attr:`input` with shape
    :math:`(N, C, H_{in}, W_{in})` and :attr:`grid` with shape
    :math:`(N, H_{out}, W_{out}, 2)`, the output will have shape
    :math:`(N, C, H_{out}, W_{out})`.

    For each output location ``output[n, :, h, w]``, the size-2 vector
    ``grid[n, h, w]`` specifies :attr:`input` pixel locations ``x`` and ``y``,
    which are used to interpolate the output value ``output[n, :, h, w]``.
    In the case of 5D inputs, ``grid[n, d, h, w]`` specifies the
    ``x``, ``y``, ``z`` pixel locations for interpolating
    ``output[n, :, d, h, w]``. :attr:`mode` argument specifies ``nearest`` or
    ``bilinear`` interpolation method to sample the input pixels.

    :attr:`grid` specifies the sampling pixel locations normalized by the
    :attr:`input` spatial dimensions. Therefore, it should have most values in
    the range of ``[-1, 1]``. For example, values ``x = -1, y = -1`` is the
    left-top pixel of :attr:`input`, and values  ``x = 1, y = 1`` is the
    right-bottom pixel of :attr:`input`.

    If :attr:`grid` has values outside the range of ``[-1, 1]``, the corresponding
    outputs are handled as defined by :attr:`padding_mode`. Options are

        * ``padding_mode="zeros"``: use ``0`` for out-of-bound grid locations,
        * ``padding_mode="border"``: use border values for out-of-bound grid locations,
        * ``padding_mode="reflection"``: use values at locations reflected by
          the border for out-of-bound grid locations. For location far away
          from the border, it will keep being reflected until becoming in bound,
          e.g., (normalized) pixel location ``x = -3.5`` reflects by border ``-1``
          and becomes ``x' = 1.5``, then reflects by border ``1`` and becomes
          ``x'' = -0.5``.

    Note:
        This function is often used in conjunction with :func:`affine_grid`
        to build `Spatial Transformer Networks`_ .

    Note:
        NaN values in :attr:`grid` would be interpreted as ``-1``.

    Args:
        input (Tensor): input of shape :math:`(N, C, H_{in}, W_{in})` (4-D case)
                        or :math:`(N, C, D_{in}, H_{in}, W_{in})` (5-D case)
        grid (Tensor): flow-field of shape :math:`(N, H_{out}, W_{out}, 2)` (4-D case)
                       or :math:`(N, D_{out}, H_{out}, W_{out}, 3)` (5-D case)
        mode (str): interpolation mode to calculate output values
            ``'bilinear'`` | ``'nearest'`` | ``'bicubic'``. Default: ``'bilinear'``
            Note: ``mode='bicubic'`` supports only 4-D input.
            When ``mode='bilinear'`` and the input is 5-D, the interpolation mode
            used internally will actually be trilinear. However, when the input is 4-D,
            the interpolation mode will legitimately be bilinear.
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        align_corners (bool): Geometrically, we consider the pixels of the
            input  as squares rather than points.
            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
            to the center points of the input's corner pixels. If set to ``False``, they
            are instead considered as referring to the corner points of the input's corner
            pixels, making the sampling more resolution agnostic.
            This option parallels the ``align_corners`` option in
            :func:`interpolate`, and so whichever option is used here
            should also be used there to resize the input image before grid sampling.
            Default: ``False``

    Returns:
        output (Tensor): output Tensor

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    .. note::
        ``mode='bicubic'`` is implemented using the `cubic convolution algorithm`_ with :math:`\\alpha=-0.75`.
        The constant :math:`\\alpha` might be different from packages to packages.
        For example, `PIL`_ and `OpenCV`_ use -0.5 and -0.75 respectively.
        This algorithm may "overshoot" the range of values it's interpolating.
        For example, it may produce negative values or values greater than 255 when interpolating input in [0, 255].
        Clamp the results with :func: `flow.clamp` to ensure they are within the valid range.
    .. _`cubic convolution algorithm`: https://en.wikipedia.org/wiki/Bicubic_interpolation
    .. _`PIL`: https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/src/libImaging/Resample.c#L51
    .. _`OpenCV`: https://github.com/opencv/opencv/blob/f345ed564a06178670750bad59526cfa4033be55/modules/imgproc/src/resize.cpp#L908
    
    Examples::

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.arange(1., 11).reshape((1, 1, 2, 5)), dtype=flow.float32)
        >>> np_grid = np.array(
        ...     [[[-0.9, -4.1], [0, 0.2000], [1, -1], [-0.333, 1e-6], [0.5, 1.0]],
        ...      [[-1.0, -0.5], [0, 0.3333], [1, -1], [-0.200, 1e-6], [1.5, 0.5]]]
        ... ).reshape(1, 2, 5, 2)
        >>> grid = flow.tensor(np_grid, dtype=flow.float32)
        >>> output = flow.nn.functional.grid_sample(input, grid, mode='nearest', padding_mode='zeros',
        ...                                        align_corners=True)
        >>> output
        tensor([[[[0., 8., 5., 7., 9.],
                  [1., 8., 5., 8., 0.]]]], dtype=oneflow.float32)
    """
    y = flow._C.grid_sample(
        input,
        grid,
        interpolation_mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return y


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
