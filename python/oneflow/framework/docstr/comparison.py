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
    oneflow.greater,
    """Returns the truth value of :math:`input > other` element-wise.

    Args:
        input (oneflow.Tensor): A Tensor
        other (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: A Tensor with bool type.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> input2 = flow.tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)

        >>> out = flow.gt(input1, input2).shape
        >>> out
        oneflow.Size([2, 6, 5, 3])

    """,
)

add_docstr(
    oneflow.greater_equal,
    """Returns the truth value of :math:`input >= other` element-wise.

    Args:
        input (oneflow.Tensor): A Tensor
        other (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: A Tensor with bool type.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.tensor(np.array([1, 1, 4]).astype(np.float32), dtype=flow.float32)

        >>> out = flow.ge(input1, input2)
        >>> out
        tensor([ True,  True, False], dtype=oneflow.bool)

    """,
)

add_docstr(
    oneflow.eq,
    """eq(input, other) -> Tensor

    Computes element-wise equality.
    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.

    Args:
        input (oneflow.Tensor): the tensor to compare
        other (oneflow.Tensor, float or int): the target to compare

    Returns:

        - A boolean tensor that is True where :attr:`input` is equal to :attr:`other` and False elsewhere

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.tensor(np.array([2, 3, 4, 5]), dtype=flow.float32)
        >>> other = flow.tensor(np.array([2, 3, 4, 1]), dtype=flow.float32)

        >>> y = flow.eq(input, other)
        >>> y
        tensor([ True,  True,  True, False], dtype=oneflow.bool)

    """,
)

add_docstr(
    oneflow.equal,
    """equal(input, other) -> bool

    `True` if two tensors have the same size and elements, `False` otherwise.

    Args:
        input (oneflow.Tensor): the tensor to compare
        other (oneflow.Tensor): the target to compare

    Returns:
        A boolean value

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.tensor(np.array([2, 3, 4, 5]), dtype=flow.float32)
        >>> other = flow.tensor(np.array([2, 3, 4, 1]), dtype=flow.float32)

        >>> y = flow.equal(input, other)
        >>> y
        False

        >>> y = flow.equal(input, input)
        >>> y
        True

    """,
)

add_docstr(
    oneflow.ne,
    """ne(input, other) -> Tensor

    Computes element-wise not equality.
    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.

    Args:
        input (oneflow.Tensor): the tensor to compare
        other (oneflow.Tensor, float or int): the target to compare

    Returns:

        - A boolean tensor that is True where :attr:`input` is not equal to :attr:`other` and False elsewhere

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.tensor(np.array([2, 3, 4, 5]), dtype=flow.float32)
        >>> other = flow.tensor(np.array([2, 3, 4, 1]), dtype=flow.float32)

        >>> y = flow.ne(input, other)
        >>> y
        tensor([False, False, False,  True], dtype=oneflow.bool)

    """,
)

add_docstr(
    oneflow.lt,
    """lt(input, other) -> Tensor

    Returns the truth value of :math:`input < other` element-wise.

    Args:
        input (oneflow.Tensor): A Tensor
        other (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: A Tensor with bool type.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.tensor(np.array([1, 2, 4]).astype(np.float32), dtype=flow.float32)

        >>> out = flow.lt(input1, input2)
        >>> out
        tensor([False, False,  True], dtype=oneflow.bool)

    """,
)


add_docstr(
    oneflow.le,
    """le(input, other) -> Tensor
    
    Returns the truth value of :math:`input <= other` element-wise.

    Args:
        input (oneflow.Tensor): A Tensor
        other (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: A Tensor with bool type.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.tensor(np.array([1, 1, 4]).astype(np.float32), dtype=flow.float32)

        >>> out = flow.le(input1, input2)
        >>> out
        tensor([ True, False,  True], dtype=oneflow.bool)

    """,
)

add_docstr(
    oneflow.isclose,
    r"""isclose(input, other, atol=1e-08, rtol=1e-05, equal_nan=False) -> Tensor
    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.isclose.html

    Returns a new tensor with boolean elements representing if each element of
    :attr:`input` is "close" to the corresponding element of :attr:`other`.
    Closeness is defined as:

    .. math::
        \lvert \text{input} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert

    Args:
        input (oneflow.Tensor): first tensor to compare
        other (oneflow.Tensor): second tensor to compare
        atol (float, optional): absolute tolerance. Default: 1e-08
        rtol (float, optional): relative tolerance. Default: 1e-05
        equal_nan (bool, optional): if ``True``, then two ``NaN`` s will be considered equal. Default: ``False``

    Returns:
        oneflow.Tensor: A Tensor with bool type.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> flow.isclose(flow.tensor((1., 2, 3)), flow.tensor((1 + 1e-10, 3, 4)))
        tensor([ True, False, False], dtype=oneflow.bool)

        >>> flow.isclose(flow.tensor((float('inf'), 4)), flow.tensor((float('inf'), 6)), rtol=.5)
        tensor([True, True], dtype=oneflow.bool)

    """,
)

add_docstr(
    oneflow.allclose,
    r"""allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool
    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.allclose.html

    This function checks if :attr:`input` and :attr:`other` satisfy the condition:

    .. math::
        \lvert \text{input} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert

    elementwise, for all elements of :attr:`input` and :attr:`other`. The behaviour of this function is analogous to
    `numpy.allclose <https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html>`_

    Args:
        input (oneflow.Tensor): first tensor to compare
        other (oneflow.Tensor): second tensor to compare
        atol (float, optional): absolute tolerance. Default: 1e-08
        rtol (float, optional): relative tolerance. Default: 1e-05
        equal_nan (bool, optional): if ``True``, then two ``NaN`` s will be considered equal. Default: ``False``

    Returns:
        oneflow.Tensor: A Tensor with bool type.

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> flow.allclose(flow.tensor([10000., 1e-07]), flow.tensor([10000.1, 1e-08]))
        False
        >>> flow.allclose(flow.tensor([10000., 1e-08]), flow.tensor([10000.1, 1e-09]))
        True
        >>> flow.allclose(flow.tensor([1.0, float('nan')]), flow.tensor([1.0, float('nan')]))
        False
        >>> flow.allclose(flow.tensor([1.0, float('nan')]), flow.tensor([1.0, float('nan')]), equal_nan=True)
        True

    """,
)
