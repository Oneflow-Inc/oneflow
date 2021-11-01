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


def l2_normalize(input, dim=0, epsilon=1e-12):
    """Use L2 norm to normalizes along dimension `dim`

    The equation is:

    .. math::
        out = \\frac{x}{max(\\sqrt{\\Sigma{x^2}}, \\epsilon)}

    Args:
        input (oneflow.Tensor): Input Tensor
        dim (int): The axis on which to apply L2 normalization. Defaults to 0.
        epsilon (float, optional): The epsilon value is used to avoid division by zero. Defaults to 1e-12.

    Returns:
        oneflow.Tensor: The normalized Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([[1, 2], [3, 4]], dtype=flow.float32)
        >>> out = flow.nn.functional.l2_normalize(x, 0)
        >>> out
        tensor([[0.3162, 0.4472],
                [0.9487, 0.8944]], dtype=oneflow.float32)
        >>> out = flow.nn.functional.l2_normalize(x, 1)
        >>> out
        tensor([[0.4472, 0.8944],
                [0.6000, 0.8000]], dtype=oneflow.float32)

    """
    y, _ = flow._C.l2_normalize(input, dim, epsilon)
    return y


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
