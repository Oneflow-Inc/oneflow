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
from cgitb import reset
from typing import List, Optional, Union
import math
import oneflow as flow


def logspace_op(
    start: float,
    end: float,
    steps: int,
    base: Optional[float] = 10.0,
    dtype: flow.dtype = None,
    device: Union[str, flow.device] = None,
    placement: flow.placement = None,
    sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    requires_grad: bool = False,
):
    r"""
    logspace(start, end, steps, base=10.0, *, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    This function is equivalent to PyTorch’s logspace function. 
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.logspace.html.

    Creates a one-dimensional tensor of size :attr:`steps` whose values are evenly
    spaced from :math:`{{\text{{base}}}}^{{\text{{start}}}}` to
    :math:`{{\text{{base}}}}^{{\text{{end}}}}`, inclusive, on a logarithmic scale
    with base :attr:`base`. That is, the values are:

    .. math::
        (\text{base}^{\text{start}},
        \text{base}^{(\text{start} + \frac{\text{end} - \text{start}}{ \text{steps} - 1})},
        \ldots,
        \text{base}^{(\text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{ \text{steps} - 1})},
        \text{base}^{\text{end}})

    Args:
        start (float): the starting value for the set of points
        end (float): the ending value for the set of points
        steps (int): size of the constructed tensor
        base (float, optional): base of the logarithm function. Default: ``10.0``.

    Keyword arguments:
        dtype (oneflow.dtype, optional): the data type to perform the computation in.
            Default: if None, uses the global default dtype (see oneflow.get_default_dtype())
            when both :attr:`start` and :attr:`end` are real,
            and corresponding complex dtype when either is complex.
        device (oneflow.device, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor type
        placement (oneflow.placement, optional): the desired placement of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        sbp (oneflow.sbp.sbp or tuple of oneflow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    Example::

        >>> import oneflow as flow
        >>> flow.logspace(start=-10, end=10, steps=2)
        tensor([1.0000e-10, 1.0000e+10], dtype=oneflow.float32)
        >>> flow.logspace(start=0.1, end=1.0, steps=5)
        tensor([ 1.2589,  2.1135,  3.5481,  5.9566, 10.0000], dtype=oneflow.float32)
        >>> flow.logspace(start=0.1, end=1.0, steps=1)
        tensor([1.2589], dtype=oneflow.float32)
        >>> flow.logspace(start=2, end=2, steps=1, base=2)
        tensor([4.], dtype=oneflow.float32)

    """
    # TODO: Migrate to C++
    indice = flow.linspace(
        start=start,
        end=end,
        steps=steps,
        dtype=dtype,
        device=device,
        placement=placement,
        sbp=sbp,
    )
    res = flow.pow(base, indice)
    res.requires_grad = requires_grad
    return res
