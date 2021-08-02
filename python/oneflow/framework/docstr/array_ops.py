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
    oneflow.F.avg_pool_2d,
    r"""
    avg_pool_2d(x: Tensor, *, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: string, padding_before: Union[int, Tuple[int, int]], padding_after: Union[int, Tuple[int, int]], data_format="channels_first": string, ceil_mode=False: bool) -> Tensor

    Applies 2D average-pooling operation in kH x kW regions by step size sH x sW steps. The number of output features is equal to the number of input planes.
    
    See :func:`oneflow.nn.AvgPool2d` for details and output shape. 
    
    Args:
        x (Tensor): the input tensor.
        kernel_size (Union[int, Tuple[int, int]]):  an int or list of ints that has length 1, 2. The size of the window for each dimension of the input Tensor.
        stride (Union[int, Tuple[int, int]]): an int or list of ints that has length 1, 2. The stride of the sliding window for each dimension of the input Tensor.
        padding (string): the padding type. "valid" adds no zero padding. "same_lower" adds padding the low position such that if the stride is 1, the output shape is the same as input shape. "same_upper" adds padding the upper position such that if the stride is 1, the output shape is the same as input shape. "customized" adds padding in the customized type.
        padding_before (Union[int, Tuple[int, int]]): the padding elements.
        padding_after (Union[int, Tuple[int, int]]): the padding elements.
        data_format (string, default to "channels_first"): "channels_first" indicates the input shape `NCHW`. "channels_last" indicates the input shape `NHWC`.
        ceil_mode (bool, default to False): When True, will use ceil instead of floor to compute the output shape.

""",
)

add_docstr(
    oneflow.F.clip_by_scalar,
    r"""
    clip_by_scalar(x: Tensor, *, min: Union[float, int], max: Union[float, int]) -> Tensor

    Clips all elements in :attr:`x` into the range [:attr:`min`, :attr:`max`]. Letting min_value and max_value be :attr:`min` and :attr:`max`, respectively, this returns:
    
    .. math::
        y_{i} = \min(\max(x_{i} ,\text{min_value}), \text{max_value})
    
    Args:
        x (Tensor): the input tensor.
        min (Union[float, int]): lower-bound of the range to be cliped to.
        max (Union[float, int]): upper-bound of the range to be cliped to.
    
    For example:
    
    .. code-block:: python
        
        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor([1., 2., 3., 4., 5.], dtype=flow.float32)
        >>> y = flow.F.clip_by_scalar(x, 2, 3)
        >>> y
        tensor([2., 2., 3., 3., 3.], dtype=oneflow.float32)

""",
)

add_docstr(
    oneflow.F.clip_by_scalar_max,
    r"""
    clip_by_scalar_max(x: Tensor, *, max: Union[float, int]) -> Tensor

    Clips all elements in :attr:`x` into the range [:math:`-\infty`, :attr:`max`]. Letting max_value be :attr:`max`, this returns:
    
    .. math::
        y_i = \min(x_i, \text{max_value})
    
    Args:
        x (Tensor): the input tensor.
        max (Union[float, int]): upper-bound of the range to be cliped to.
    
    For example:
    
    .. code-block:: python
        
        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor([1., 2., 3., 4., 5.], dtype=flow.float32)
        >>> y = flow.F.clip_by_scalar_max(x, 3)
        >>> y
        tensor([1., 2., 3., 3., 3.], dtype=oneflow.float32)

""",
)

add_docstr(
    oneflow.F.clip_by_scalar_min,
    r"""
    clip_by_scalar_min(x: Tensor, *, min: Union[float, int]) -> Tensor

    Clips all elements in :attr:`x` into the range [:attr:`min`, :math:`+\infty`]. Letting min_value be :attr:`min`, this returns:

    .. math::
        y_i = \max(x_i, \text{min_value})
    
    Args:
        x (Tensor): the input tensor.
        min (Union[float, int]): lower-bound of the range to be cliped to.
    
    For example:
    
    .. code-block:: python
        
        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor([1., 2., 3., 4., 5.], dtype=flow.float32)
        >>> y = flow.F.clip_by_scalar_min(x, 3)
        >>> y
        tensor([3., 3., 3., 4., 5.], dtype=oneflow.float32)
""",
)
