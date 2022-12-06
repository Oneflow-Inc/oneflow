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
    oneflow.logaddexp,
    """
    logaddexp(input, other, *, out=None) -> Tensor

    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.logaddexp.html.

    Logarithm of the sum of exponentiations of the inputs.

    Calculates pointwise :math:`\log\left(e^x + e^y\right)`. This function is useful
    in statistics where the calculated probabilities of events may be so small as to
    exceed the range of normal floating point numbers. In such cases the logarithm
    of the calculated probability is stored. This function allows adding
    probabilities stored in such a fashion.

    Args:
        input (oneflow.Tensor): the input Tensor.
        other (oneflow.Tensor): the second input Tensor.
        out (oneflow.Tensor, optional): the output Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> flow.logaddexp(flow.tensor([-1.0]), flow.tensor([-1.0, -2, -3]))
        tensor([-0.3069, -0.6867, -0.8731], dtype=oneflow.float32)
        >>> flow.logaddexp(flow.tensor([-100.0, -200, -300]), flow.tensor([-1.0, -2, -3]))
        tensor([-1., -2., -3.], dtype=oneflow.float32)
 
    """,
)
