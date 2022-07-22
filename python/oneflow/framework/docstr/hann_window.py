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
    oneflow.hann_window,
    r"""
    hann_window(window_length, periodic=True, *, device=None,  placement=None, sbp=None, dtype=None, requires_grad=False) -> Tensor

    This function is equivalent to PyTorch’s hann_window function. 
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.hann_window.html.

    Hann window function.

    .. math::
        w[n] = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{N - 1} \right)\right] =
                \sin^2 \left( \frac{\pi n}{N - 1} \right),

    where :math:`N` is the full window size.

    The input :attr:`window_length` is a positive integer controlling the
    returned window size. :attr:`periodic` flag determines whether the returned
    window trims off the last duplicate value from the symmetric window. Therefore, if :attr:`periodic` is true, the :math:`N` in
    above formula is in fact :math:`\text{window_length} + 1`. Also, we always have
    ``oneflow.hann_window(L, periodic=True)`` equal to
    ``oneflow.hann_window(L + 1, periodic=False)[:-1])``.

    .. note::
        If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.

    Arguments:
        window_length (int): the size of returned window
        periodic (bool, optional): If True, returns a window to be used as periodic
            function. If False, return a symmetric window.

    Keyword args:
        dtype (oneflow.dtype, optional): the data type to perform the computation in.
            Default: if None, uses the global default dtype (see oneflow.get_default_dtype())
            when both :attr:`start` and :attr:`end` are real,
            and corresponding complex dtype when either is complex.
        device (oneflow.device, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor type
        placement (oneflow.placement, optional): the desired placement of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        sbp (oneflow.sbp.sbp or tuple of oneflow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    Returns:
        Tensor: A 1-D tensor of size :math:`(\text{{window_length}},)` containing the window

    """,
)
