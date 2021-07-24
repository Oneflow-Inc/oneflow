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
from oneflow.nn.module import Module


class Softplus(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.softplus(x)


@register_tensor_op("softplus")
def softplus_op(x):
    """Applies the element-wise function:

    .. math::
        Softplus(x)= \\frac{1}{β}*log(1+exp(β∗x))

    SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive.

    For numerical stability the implementation reverts to the linear function when :attr:`input X β > threshold`.

    Args:
        beta:the value for the Softplus formulation.Default:1

        threshold:values above this revert to a linear function.Default:20

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x1 = flow.Tensor(np.array([1, 2, 3]))
        >>> x2 = flow.Tensor(np.array([1.53123589,0.54242598,0.15117185]))
        >>> x3 = flow.Tensor(np.array([1,0,-1]))

        >>> flow.softplus(x1).numpy()
        array([1.3132616, 2.126928 , 3.0485873], dtype=float32)
        >>> flow.softplus(x2).numpy()
        array([1.7270232, 1.0006962, 0.771587 ], dtype=float32)
        >>> flow.softplus(x3).numpy()
        array([1.3132616 , 0.6931472 , 0.31326166], dtype=float32)

    """
    return Softplus()(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
