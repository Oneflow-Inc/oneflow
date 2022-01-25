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
    oneflow.masked_fill,
    """
    Fills elements of :attr:`self` tensor with :attr:`value` where :attr:`mask` is True.
    The shape of :attr:`mask` must be broadcastable with the shape of the underlying tensor.

    Args:
        mask (BoolTensor): the boolean mask
        value (float): the value to fill in with

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> in_arr = np.array(
        ...     [[[-0.13169311,  0.97277078,  1.23305363,  1.56752789],
        ...     [-1.51954275,  1.87629473, -0.53301206,  0.53006478],
        ...     [-1.38244183, -2.63448052,  1.30845795, -0.67144869]],
        ...     [[ 0.41502161,  0.14452418,  0.38968   , -1.76905653],
        ...     [ 0.34675095, -0.7050969 , -0.7647731 , -0.73233418],
        ...     [-1.90089858,  0.01262963,  0.74693893,  0.57132389]]]
        ... )
        >>> fill_value = 8.7654321 # random value e.g. -1e9 3.1415
        >>> input = flow.tensor(in_arr, dtype=flow.float32)
        >>> mask = flow.tensor((in_arr > 0).astype(np.int8), dtype=flow.int)
        >>> output = flow.masked_fill(input, mask, fill_value)

        # tensor([[[-0.1317,  8.7654,  8.7654,  8.7654],
        #  [-1.5195,  8.7654, -0.533 ,  8.7654],
        #  [-1.3824, -2.6345,  8.7654, -0.6714]],

        # [[ 8.7654,  8.7654,  8.7654, -1.7691],
        #  [ 8.7654, -0.7051, -0.7648, -0.7323],
        #  [-1.9009,  8.7654,  8.7654,  8.7654]]], dtype=oneflow.float32)

    """,
)
