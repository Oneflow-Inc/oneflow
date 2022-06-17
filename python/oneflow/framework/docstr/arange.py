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
    oneflow.arange,
    """
    oneflow.arange(start: int = 0, end, step: int = 1, dtype: Optional[oneflow._oneflow_internal.dtype] = None, device: Optional[Union[oneflow._oneflow_internal.device, str]] = None, placement: Optional[oneflow._oneflow_internal.placement] = None, sbp: Optional[Union[oneflow._oneflow_internal.sbp.sbp, List[oneflow._oneflow_internal.sbp.sbp]]] = None, requires_grad: bool = False)

    Returns a 1-D tensor of size :math:`\\left\\lfloor \\frac{\\text{end} - \\text{start}}{\\text{step}} \\right\\rfloor + 1`
    with values from :attr:`start` to :attr:`end` with step :attr:`step`. Step is
    the gap between two values in the tensor.

    .. math::
        \\text{out}_{i+1} = \\text{out}_i + \\text{step}.

    Args:
        start (int): the starting value for the set of points. Default: ``0``.
        end (int): the ending value for the set of points
        step (int): the gap between each pair of adjacent points. Default: ``1``.

    Keyword args:
        dtype(flow.dtype, optional): If `dtype` is not given, infer the `dtype` from the other input arguments. If any of start, end, or step are floating-point, the `dtype` is inferred to be the floating-point data type. Otherwise, the `dtype` is inferred to be `flow.int64`.
        device(flow.device, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: `False`.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> y = flow.arange(0, 5)
        >>> y
        tensor([0, 1, 2, 3, 4], dtype=oneflow.int64)

    """,
)
