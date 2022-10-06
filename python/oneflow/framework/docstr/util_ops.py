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
    oneflow.isnan,
    """
    isnan(input) -> Tensor 
    
    This function is equivalent to PyTorch’s isnan function. 
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.isnan.html?highlight=isnan#torch.isnan

    Returns a new tensor with boolean elements representing if each element of input is NaN or not.

    Args:
        input(Tensor): the input tensor.

    Returns:
        A boolean tensor that is True where input is NaN and False elsewhere.

    Example::

        >>> import oneflow as flow
        >>> flow.isnan(flow.tensor([1, float('nan'), 2]))
        tensor([False,  True, False], dtype=oneflow.bool)

    """,
)

add_docstr(
    oneflow.isinf,
    """
    isinf(input) -> Tensor 

    This function is equivalent to PyTorch’s isinf function. 
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.isinf.html?highlight=isinf#torch.isinf

    Tests if each element of input is infinite (positive or negative infinity) or not.

    Args:
        input(Tensor): the input tensor.

    Returns:
        A boolean tensor that is True where input is infinite and False elsewhere.

    Example::

        >>> import oneflow as flow
        >>> flow.isinf(flow.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
        tensor([False,  True, False,  True, False], dtype=oneflow.bool)

    """,
)

add_docstr(
    oneflow.isfinite,
    """
    isfinite(input) -> Tensor 

    This function is equivalent to PyTorch’s isfinite function. 
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.isfinite.html?highlight=isfinite#torch.isfinite

    Returns a new tensor with boolean elements representing if each element is finite or not.

    Args:
        input(Tensor): the input tensor.

    Returns:
        A boolean tensor that is True where input is finite and False elsewhere.

    Example::

        >>> import oneflow as flow
        >>> flow.isfinite(flow.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
        tensor([ True, False,  True, False, False], dtype=oneflow.bool)

    """,
)
