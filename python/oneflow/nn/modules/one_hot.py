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


def one_hot(
    input, 
    num_classes=-1,
):
    """This operator generates a onehot Tensor from input Tensor.

    If input Tensor's rank is `N`, the corresponding onehot Tensor's rank is `N+1`.

    Args:
        input (Tensor): The input Tensor.
        num_classes (int): The length of onehot Tensor.
         
    Note:

        The data type of input blob should be `int32` or `int64`.

    Returns:
        oneflow.Tensor.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input=flow.Tensor(np.array([0, 3, 1, 2]).astype(np.int32), dtype=flow.int64)
        >>> out = flow.nn.functional.one_hot(input, num_classes=5)
        >>> out
        tensor([[1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0]], dtype=oneflow.int64)
    
    """
    if input.is_consistent:
        raise ValueError(
            "A consistent tensor can not be applied to one_hot."
        )
    if input.is_lazy:
        raise NotImplementedError
    else:
        if num_classes == -1:
            num_classes = (flow.max(input) + 1).numpy()
    return flow.F.one_hot(input, num_classes)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
