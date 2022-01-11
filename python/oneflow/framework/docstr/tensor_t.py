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
    oneflow.t,
    """
    oneflow.t(input) → Tensor.

        Expects `input` to be tensor of any dimension and reverses its dimensions. 

        0-D and 1-D tensors are returned as is. When input is a 2-D tensor this is equivalent to `transpose(input, 0, 1)`.

    Args:
        input (oneflow.Tensor): An input tensor.   
 
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.randn()
        >>> x
        tensor(-0.2048, dtype=oneflow.float32)
        >>> flow.t(x)
        tensor(-0.2048, dtype=oneflow.float32)
        >>> x = flow.randn(3)
        >>> x
        tensor([ 0.5034, -0.4999,  0.2721], dtype=oneflow.float32)
        >>> flow.t(x)
        tensor([ 0.5034, -0.4999,  0.2721], dtype=oneflow.float32)
        >>> x = flow.randn(2,3)
        >>> x
        tensor([[ 0.1939,  0.6988,  1.0040],
                [-0.2530, -1.5002,  0.1415]], dtype=oneflow.float32)
        >>> y = flow.t(x)
        >>> y
        tensor([[ 0.1939, -0.2530],
                [ 0.6988, -1.5002],
                [ 1.0040,  0.1415]], dtype=oneflow.float32)
        >>> y.shape
        oneflow.Size([3, 2])
        >>> x = flow.randn(2,3,4)
        >>> x
        tensor([[[ 1.9923e+00,  3.4746e-01, -1.5605e+00,  3.6243e-01],
                 [-6.6592e-01, -1.3315e+00, -5.0649e-01,  1.1382e+00],
                 [ 4.2284e-01, -3.3976e-01, -1.1531e+00, -5.9660e-02]],

                [[ 5.0436e-04,  4.3516e-01, -1.7847e+00, -1.8923e-01],
                 [ 3.8091e-01, -2.0200e-01,  6.2334e-02, -3.0588e-01],
                 [ 4.0383e-01,  4.8231e-01, -9.7060e-01,  2.4865e-01]]], dtype=oneflow.float32) 
        >>> y = flow.t(x)
        >>> y
        tensor([[[ 1.9923e+00,  5.0436e-04],
                 [-6.6592e-01,  3.8091e-01],
                 [ 4.2284e-01,  4.0383e-01]],

                [[ 3.4746e-01,  4.3516e-01],
                 [-1.3315e+00, -2.0200e-01],
                 [-3.3976e-01,  4.8231e-01]],

                [[-1.5605e+00, -1.7847e+00],
                 [-5.0649e-01,  6.2334e-02],
                 [-1.1531e+00, -9.7060e-01]],

                [[ 3.6243e-01, -1.8923e-01],
                 [ 1.1382e+00, -3.0588e-01],
                 [-5.9660e-02,  2.4865e-01]]], dtype=oneflow.float32) 
        >>> y.shape
        oneflow.Size([4, 3, 2]) 
    
    """,
)
