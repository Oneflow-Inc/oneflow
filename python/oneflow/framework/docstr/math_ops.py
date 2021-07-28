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
    "\n    sin(x: Tensor) -> Tensor\n\n    Returns a new tensor with the sine of the elements of :attr:`input`.\n\n    .. math::\n\n        \\text{y}_{i} = \\sin(\\text{x}_{i})\n\n    Args:\n        x (Tensor): the input tensor.\n\n    For example:\n\n    .. code-block:: python\n\n        >>> import oneflow as flow\n        >>> import numpy as np\n        >>> x1 = flow.Tensor(np.array([-0.5461,  0.1347, -2.7266, -0.2746]).astype(np.float32))\n        >>> y1 = flow.F.sin(x1)\n        >>> y1\n        tensor([-0.5194,  0.1343, -0.4032, -0.2712], dtype=oneflow.float32)\n        >>> x2 = flow.Tensor(np.array([-1.4, 2.6, 3.7]).astype(np.float32),device=flow.device('cuda'))\n        >>> y2 = flow.F.sin(x2)\n        >>> y2\n        tensor([-0.9854,  0.5155, -0.5298], device='cuda:0', dtype=oneflow.float32)\n\n\n",
)
add_docstr(
    oneflow.F.cos,
    "\n    cos(x: Tensor) -> Tensor\n\n    Returns a new tensor with the cosine  of the elements of :attr:`input`.\n    \n    .. math::\n        \\text{y}_{i} = \\cos(\\text{x}_{i})\n\n    Args:\n        x (Tensor): the input tensor.\n\n    For example:\n\n    .. code-block:: python\n\n        >>> import oneflow as flow\n        >>> import numpy as np\n        >>> x = np.array([1.4309,  1.2706, -0.8562,  0.9796])\n        >>> x = flow.Tensor(x, dtype=flow.float32)\n        >>> y = flow.F.cos(x)\n        >>> y\n        tensor([0.1394, 0.2957, 0.6553, 0.5574], dtype=oneflow.float32)\n\n",
)

add_docstr(
    oneflow.F.acos,
    r"""
    acos(x: Tensor) -> Tensor

    Returns a new tensor with the inverse cosine of the elements of :attr:`input`.
    
    .. math::
        \text{out}_{i} = \arccos(\text{input}_{i})
        
    Args:
        input (Tensor): the input tensor.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.array([0.5, 0.6, 0.7])
        >>> input = flow.Tensor(arr, dtype=flow.float32)
        >>> output = flow.F.acos(input)
        >>> output
        tensor([1.0472, 0.9273, 0.7954], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.F.add,
    r"""
    add(Tensor x, Tensor y, *, Bool inplace=False) -> Tensor

    Each element of the tensor other added to each element of the tensor input. The resulting tensor is returned.
    
    .. math::
        \text{out} = \text{input}+\text{other}
        
    Args:
        input (Tensor): the input tensor.

        other (Number): the other tensor.
    
    For example:

    .. code-block:: python
        
        >>> import oneflow as flow
        >>> a = flow.Tensor([1., 2., 3.])
        >>> a
        tensor([1., 2., 3.], dtype=oneflow.float32)
        >>> b = flow.Tensor([4., 5., 6.])
        >>> b
        tensor([4., 5., 6.], dtype=oneflow.float32)
        >>> out = flow.F.add(a, b)
        >>> out
        tensor([5., 7., 9.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.F.add_n,
    r"""
    add_n(TensorTuple inputs, *, Bool inplace=False) -> Tensor

    Adds all input tensors element-wise.

    Returns:
        A flow.Tensor of the same shape and type as the elements of inputs.
        
    Args:
        input (Tensor): A tuple of flow.Tensor, each with the same shape and type.

        inplace (Bool): If set to True, will do this operation in-place. Default: False
    
    For example:

    .. code-block:: python
        
        >>> import oneflow as flow
        >>> a = flow.tensor([1.0, 2.0, 3.0])
        >>> a
        tensor([1., 2., 3.], dtype=oneflow.float32)
        >>> b = flow.tensor([2.0, 3.0, 4.0])
        >>> b
        tensor([2., 3., 4.], dtype=oneflow.float32)
        >>> flow.F.add_n((a,b))
        tensor([3., 5., 7.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.F.add_scalar,
    r"""
    add_scalar(Tensor x, *, Scalar alpha, Bool inplace=False) -> Tensor

    Adds the scalar other to each element of the input input and returns a new resulting tensor.

    Args:
        x (oneflow.Tensor): Input Tensor.
        Scalar alpha: The scalar.
    
    For example:

    .. code-block:: python
        
        >>> import oneflow as flow
        >>> x = flow.Tensor([1., 3., 4., 5., 6.])
        >>> x
        tensor([1., 3., 4., 5., 6.], dtype=oneflow.float32)
        >>> flow.F.add_scalar(x, 2.1)
        tensor([3.1, 5.1, 6.1, 7.1, 8.1], dtype=oneflow.float32)

    """
)

add_docstr(
    oneflow.F.argmax,
    r"""
    argmax(Tensor x) -> Tensor
        
    The op computes the index with the largest value of a Tensor at 0 axis.
        
    Returns:
        Returns the indices of the maximum value of all elements in the input tensor.

    Args:
        x (oneflow.Tensor): Input Tensor.
        
    For example:

    .. code-block:: python
        
        >>> import oneflow as flow  
        >>> x = flow.Tensor([[1, 3, 8, 9, 2],
        ...                  [1, 7, 4, 3, 2]])
        >>> out = flow.F.argmax(x)
        >>> out
        tensor([3, 1], dtype=oneflow.int32)

    """,
)

add_docstr(
    oneflow.F.asin,
    r"""
    asin(Tensor x) -> Tensor
                
    Returns:
        A new tensor with the arcsine of the elements of input.

    .. math::
        \text{out}_{i} = \sin^{-1}(\text{input}_{i})

    Args:
        x (oneflow.Tensor): Input Tensor.
        
    For example:

    .. code-block:: python
        
        >>> import oneflow as flow  
        >>> x = flow.Tensor([-0.0126,  0.1886, -0.5606,  1.0285])
        >>> x
        tensor([-0.0126,  0.1886, -0.5606,  1.0285], dtype=oneflow.float32)
        >>> flow.F.asin(x)
        tensor([-0.0126,  0.1897, -0.5951,     nan], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.F.asinh,
    r"""
    asinh(Tensor x) -> Tensor
                
    Returns:
        A new tensor with the inverse hyperbolic sine of the elements of input.

    .. math::
        \text{out}_{i} = \sinh^{-1}(\text{input}_{i})

    Args:
        x (oneflow.Tensor): Input Tensor.
        
    For example:

    .. code-block:: python
        
        >>> import oneflow as flow  
        >>> x = flow.Tensor([-0.0126,  0.1886, -0.5606,  1.0285])
        >>> x
        tensor([-0.0126,  0.1886, -0.5606,  1.0285], dtype=oneflow.float32)
        >>> flow.F.asinh(x)
        tensor([-0.0126,  0.1875, -0.5347,  0.9014], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.F.atan,
    r"""
    atan(Tensor x) -> Tensor
                
    Returns:
        a new tensor with the arctangent of the elements of input.

    .. math::
        \text{out}_{i} = \tan^{-1}(\text{input}_{i})

    Args:
        x (oneflow.Tensor): Input Tensor.
        
    For example:

    .. code-block:: python
        
        >>> import oneflow as flow  
        >>> x = flow.Tensor([-0.0126,  0.1886, -0.5606,  1.0285])
        >>> x
        tensor([-0.0126,  0.1886, -0.5606,  1.0285], dtype=oneflow.float32)
        >>> flow.F.atan(x)
        tensor([-0.0126,  0.1864, -0.5109,  0.7994], dtype=oneflow.float32)

    """,
)
