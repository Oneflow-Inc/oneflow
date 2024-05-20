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
import warnings
import oneflow as flow
from oneflow.framework.tensor import Tensor


def relu6(input: Tensor, inplace=False) -> Tensor:
    r"""relu6(input: Tensor, inplace=False) -> Tensor

    Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`.

    See :class:`~oneflow.nn.ReLU6` for more details.
    """
    if inplace:
        warnings.warn("relu6 do not support inplace now")
    return flow._C.hardtanh(input, min_val=0.0, max_val=6.0)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
