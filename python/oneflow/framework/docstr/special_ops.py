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
    oneflow.special.digamma,
    """
    Alias for :func:`oneflow.digamma`. 
    """,
)

add_docstr(
    oneflow.special.erf,
    """
    Alias for :func:`oneflow.erf`. 
    """,
)

add_docstr(
    oneflow.special.erfc,
    """
    Alias for :func:`oneflow.erfc`. 
    """,
)

add_docstr(
    oneflow.special.erfinv,
    """
    Alias for :func:`oneflow.erfinv`. 
    """,
)

add_docstr(
    oneflow.special.exp2,
    """
    Alias for :func:`oneflow.exp2`. 
    """,
)

add_docstr(
    oneflow.special.expm1,
    """
    Alias for :func:`oneflow.expm1`. 
    """,
)

add_docstr(
    oneflow.special.log1p,
    """
    Alias for :func:`oneflow.log1p`. 
    """,
)

add_docstr(
    oneflow.special.log_softmax,
    """
    Alias for :func:`oneflow.nn.functional.log_softmax`. 
    """,
)

add_docstr(
    oneflow.special.logsumexp,
    """
    Alias for :func:`oneflow.logsumexp`. 
    """,
)

add_docstr(
    oneflow.special.round,
    """
    Alias for :func:`oneflow.round`. 
    """,
)

add_docstr(
    oneflow.special.softmax,
    """
    Alias for :func:`oneflow.softmax`. 
    """,
)

add_docstr(
    oneflow.special.psi,
    """
    Alias for :func:`oneflow.special.digamma`. 
    """,
)

add_docstr(
    oneflow.special.zeta,
    r"""
    zeta(input, other) -> Tensor
    
    Computes the Hurwitz zeta function, elementwise.
    
    .. math::
        \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x}
    
    Args:
        input (Tensor): the input tensor corresponding to `x`.
        other (Tensor): the input tensor corresponding to `q`.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([2., 4.])
        >>> flow.special.zeta(x, 1)
        tensor([1.6449, 1.0823], dtype=oneflow.float32)
        >>> flow.special.zeta(x, flow.tensor([1., 2.]))
        tensor([1.6449, 0.0823], dtype=oneflow.float32)
        >>> flow.special.zeta(2,flow.tensor([1., 2.]))
        tensor([1.6449, 0.6449], dtype=oneflow.float32)

    """,
)
