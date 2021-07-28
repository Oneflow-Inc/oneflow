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
    oneflow.F.ceil,
    """
    ceil(x: Tensor) -> Tensor


    Returns a new tensor with the ceil of the elements of :attr:`x`,
    the smallest integer greater than or equal to each element.

    The equation is: 

    .. math::
        \\text{out}_{i} = \\left\\lceil \\text{input}_{i} \\right\\rceil = \\left\\lfloor \\text{input}_{i} \\right\\rfloor + 1

    Args:
        x (oneflow.Tensor): A Tensor.
    
    Returns:
        oneflow.Tensor: The result Tensor

    For example: 


    .. code-block:: python 
        
        >>> import oneflow as flow
        >>> import numpy as np   
        >>> x = flow.Tensor(np.array([0.1, -2, 3.4]).astype(np.float32))
        >>> y = flow.F.ceil(x)
        >>> print(y.shape)
        flow.Size([3])
        >>> print(y.numpy())
        [ 1. -2.  4.]


        >>> x = flow.Tensor(np.array([[[2.2, 4.4, 6.5],[7.1, 8.2, 9.3]],[[10.6,11.2,12.2],[13.5,14.8,15.9]]]).astype(np.float32))
        >>> y = flow.F.ceil(x)
        >>> print(y.shape)
        flow.Size([2, 2, 3])
        >>> print(y.numpy())
        [[[ 3.  5.  7.]
          [ 8.  9. 10.]]
        <BLANKLINE>
         [[11. 12. 13.]
          [14. 15. 16.]]]

    """,
)
add_docstr(
    oneflow.F.broadcast_add,
    r""" 
    broadcast_add(x: Tensor,y : Tensor) -> Tensor
    
    Returns a new tensor of the broadcast addition between two tensors.
    A broadcast operator process two tensors in different shapes. Normally, one of the operands has a particular dimension to be 1, which will be broadcast along the corresponding dimension of the other operator to perform the given calculation. Common scalar calculations can all be broadcast.

    The equation is: 

    .. math::
        out = x + y

    Args:
        x (oneflow.Tensor): A Tensor.
        y (oneflow.Tensor): A Tensor.
        out: the output tensor

    Returns:
        oneflow.Tensor: The result Tensor broadcast_added by x and y 

    For example: 


    .. code-block:: python 

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.array([[2], [1], [3]]).astype(np.float32))
        >>> y = flow.Tensor(np.array([[0,2,1,6], [1,0,2,1], [2,3,4,0]]).astype(np.float32))
        >>> z = flow.F.broadcast_add(x,y)
        >>> print(z.shape)
        flow.Size([3, 4])
        >>> print(z.numpy())
        [[2. 4. 3. 8.]
         [2. 1. 3. 2.]
         [5. 6. 7. 3.]]

    """,
)
