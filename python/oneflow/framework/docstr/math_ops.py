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
    oneflow.F.sin,
    r"""
    sin(x: Tensor) -> Tensor

    Returns a new tensor with the sine of the elements of :attr:`input`.

    .. math::

        \text{y}_{i} = \sin(\text{x}_{i})

    Args:
        x (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x1 = flow.Tensor(np.array([-0.5461,  0.1347, -2.7266, -0.2746]).astype(np.float32))
        >>> y1 = flow.F.sin(x1)
        >>> y1
        tensor([-0.5194,  0.1343, -0.4032, -0.2712], dtype=oneflow.float32)
        >>> x2 = flow.Tensor(np.array([-1.4, 2.6, 3.7]).astype(np.float32), device=flow.device('cuda'))
        >>> y2 = flow.F.sin(x2)
        >>> y2
        tensor([-0.9854,  0.5155, -0.5298], device='cuda:0', dtype=oneflow.float32)
    """,
)
add_docstr(
    oneflow.F.cos,
    r"""
    cos(x: Tensor) -> Tensor

    Returns a new tensor with the cosine  of the elements of :attr:`input`.
    
    .. math::
        \text{y}_{i} = \cos(\text{x}_{i})

    Args:
        x (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = np.array([1.4309,  1.2706, -0.8562,  0.9796])
        >>> x = flow.Tensor(x, dtype=flow.float32)
        >>> y = flow.F.cos(x)
        >>> y
        tensor([0.1394, 0.2957, 0.6553, 0.5574], dtype=oneflow.float32)
    
    """,
)
