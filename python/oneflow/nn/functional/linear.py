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


def linear(input, weight, bias=None):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Args:
        input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of additional dimensions
        weight: :math:`(out\_features, in\_features)`
        bias: :math:`(out\_features)`
    
    Returns:
        output: :math:`(N, *, out\_features)`

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> input = flow.tensor(np.random.randn(128, 20))
        >>> weight = flow.tensor(np.random.randn(20, 30))
        >>> output = flow.nn.functional.linear(input, weight)
        >>> output.size()
        oneflow.Size([128, 30])

    """
    res = flow._C.matmul(input, weight, transpose_a=False, transpose_b=True)
    if bias is not None:
        res += bias
    return res
