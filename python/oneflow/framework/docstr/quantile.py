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
    oneflow.quantile,
    """
    quantile(input, q, dim=None, keepdim=False, *, interpolation='linear', out=None) -> Tensor

    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.quantile.html.

    Computes the q-th quantiles of each row of the :attr:`input` tensor along the dimension :attr:`dim`.

    To compute the quantile, we map q in [0, 1] to the range of indices [0, n] to find the location
    of the quantile in the sorted input. If the quantile lies between two data points ``a < b`` with
    indices ``i`` and ``j`` in the sorted order, result is computed according to the given
    :attr:`interpolation` method as follows:

    - ``linear``: ``a + (b - a) * fraction``, where ``fraction`` is the fractional part of the computed quantile index.
    - ``lower``: ``a``.
    - ``higher``: ``b``.
    - ``nearest``: ``a`` or ``b``, whichever's index is closer to the computed quantile index (rounding down for .5 fractions).
    - ``midpoint``: ``(a + b) / 2``.

    If :attr:`q` is a 1D tensor, the first dimension of the output represents the quantiles and has size
    equal to the size of :attr:`q`, the remaining dimensions are what remains from the reduction.

    .. note::
        By default :attr:`dim` is ``None`` resulting in the :attr:`input` tensor being flattened before computation.
 
    Args:
        input (oneflow.Tensor): the input Tensor.
        q (float or oneflow.Tensor): a scalar or 1D tensor of values in the range [0, 1].
        dim (int, optional): the dimension to reduce. Default is None.
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default is False
        interpolation (str, optional): interpolation method to use when the desired quantile lies between two data points.
                                Can be ``linear``, ``lower``, ``higher``, ``midpoint`` and ``nearest``.
                                Default is ``linear``.
        out (oneflow.Tensor, optional): the output Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> a = flow.arange(8.)
        >>> q = flow.tensor([0.25, 0.5, 0.75])
        >>> flow.quantile(a, q, dim=0, keepdim=True)
        tensor([[1.7500],
                [3.5000],
                [5.2500]], dtype=oneflow.float32)
        >>> a = flow.arange(4.)
        >>> flow.quantile(a, 0.6, interpolation="linear")
        tensor(1.8000, dtype=oneflow.float32)
        >>> flow.quantile(a, 0.6, interpolation="lower")
        tensor(1., dtype=oneflow.float32)
        >>> flow.quantile(a, 0.6, interpolation="higher")
        tensor(2., dtype=oneflow.float32)
        >>> flow.quantile(a, 0.6, interpolation="midpoint")
        tensor(1.5000, dtype=oneflow.float32)
        >>> flow.quantile(a, 0.6, interpolation="nearest")
        tensor(2., dtype=oneflow.float32)
        >>> flow.quantile(a, 0.4, interpolation="nearest")
        tensor(1., dtype=oneflow.float32)

    """,
)
