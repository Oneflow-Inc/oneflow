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
    oneflow.tile,
    """
    tile(input, dims) -> Tensor

    Constructs a tensor by repeating the elements of ``input``.  The ``dims`` argument specifies the number
    of repetitions in each dimension.

    If ``dims`` specifies fewer dimensions than ``input`` has, then ones are prepended to ``dims`` until
    all dimensions are specified.  For example, if ``input`` has shape (8, 6, 4, 2) and ``dims`` is (2, 2),
    then ``dims`` is treated as (1, 1, 2, 2).

    Analogously, if ``input`` has fewer dimensions than ``dims`` specifies, then ``input`` is treated as
    if it were unsqueezed at dimension zero until it has as many dimensions as ``dims`` specifies.
    For example, if ``input`` has shape (4, 2) and ``dims`` is (3, 3, 2, 2), then ``input`` is treated as
    if it had the shape (1, 1, 4, 2).

    .. note::
        This function is similar to NumPy’s tile function.
    
    The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.tile.html.

    Args:
        input (oneflow.Tensor): the tensor whose elements to repeat.
        dims (tuple): the number of repetitions per dimension.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> np_arr = np.random.randn(5, 3, 6, 9).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> out = input.tile(2,1,2,1)
        >>> out.shape
        oneflow.Size([10, 3, 12, 9])
        >>> x = np.random.randn(5, 2, 1)
        >>> input = flow.Tensor(x)
        >>> out = input.tile(3,4)
        >>> out.shape
        oneflow.Size([5, 6, 4])
    """,
)
