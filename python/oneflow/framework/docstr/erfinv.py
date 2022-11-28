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
    oneflow.erfinv,
    """Computes the inverse error function of :attr:`input`. The inverse error function is defined in the range :math:`(-1, 1)` as:

    .. math::
        \mathrm{erfinv}(\mathrm{erf}(x)) = x

    Args:
        input (oneflow.Tensor): the input tensor.

    For example:

    .. code-block:: python
    
        >>> import oneflow as flow
        >>> import numpy as np
               
        >>> input=flow.tensor(np.random.randn(3,3).astype(np.float32))
        >>> of_out=flow.erfinv(input)
        >>> of_out.shape
        oneflow.Size([3, 3])


    """,
)
