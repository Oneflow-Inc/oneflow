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
    oneflow.addcdiv,
    r"""
    addcdiv(input, tensor1, tensor2, *, value=1) -> Tensor

    This function is equivalent to PyTorch’s addcdiv function. 
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.addcdiv.html.
    
    Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`,
    multiply the result by the scalar :attr:`value` and add it to :attr:`input`.

    .. math::
        \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{tensor1}_i}{\text{tensor2}_i}


    The shapes of :attr:`input`, :attr:`tensor1`, and :attr:`tensor2` must be
    `broadcastable`.

    For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
    a real number, otherwise an integer.

    Args:
        input (Tensor): the tensor to be added
        tensor1 (Tensor): the numerator tensor
        tensor2 (Tensor): the denominator tensor

    Keyword args:
        value (Number, optional): multiplier for :math:`\text{{tensor1}} / \text{{tensor2}}`

    Example::

        >>> import oneflow as flow
        >>> input = flow.tensor([ 0.3810,  1.2774, -0.2972, -0.3719])
        >>> tensor1 = flow.tensor([0.8032,  0.2930, -0.8113, -0.2308])
        >>> tensor2 = flow.tensor([[0.5], [1]])
        >>> output = flow.addcdiv(input, tensor1, tensor2)
        >>> output.shape
        oneflow.Size([2, 4])
    """,
)
