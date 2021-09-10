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
from oneflow.framework.tensor import register_tensor_op

def movedim_op(input, source, destination):
     """
    Moves the dimension(s) of input at the position(s) in source to the position(s) in destination.
    Other dimensions of input that are not explicitly moved remain in their original order and appear at the positions not specified in destination.

    Args:
        input (Tensor): the input tensor.
        source  (int or tuple of python:ints): Original positions of the dims to move. These must be unique. 
        destination (int or tuple of python:ints) – Destination positions for each of the original dims. These must also be unique.
    
    Returns:
        oneflow.Tensor: the output Tensor.

    For example:

    .. code-block:: python
        
        >>> import oneflow as flow
        >>> t = flow.randn(3,2,1)
        >>> t
        tensor([[[-0.1523],
                [-0.6242]],

                [[-0.1521],
                [ 0.5220]],

                [[-0.2166],
                [ 1.5300]]], dtype=oneflow.float32)
        >>> flow.movedim(t,1,0).shape
        flow.Size([2, 3, 1])
        >>> flow.movedim(t,1,0)
        tensor([[[-0.1523],
                [-0.1521],
                [-0.2166]],

                [[-0.6242],
                [ 0.5220],
                [ 1.5300]]], dtype=oneflow.float32)
        >>> flow.movedim(t, (1, 2), (0, 1)).shape
        flow.Size([2, 1, 3])
        >>> flow.movedim(t, (1, 2), (0, 1))
        tensor([[[-0.1523, -0.1521, -0.2166]],

                [[-0.6242,  0.5220,  1.5300]]], dtype=oneflow.float32)
    """
     return flow._C.movedim(input, source, destination)

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)