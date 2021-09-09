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
from typing import Union
import oneflow as flow


def one_hot(
    input,
    num_classes: int = -1,
    on_value: Union[int, float] = 1,
    off_value: Union[int, float] = 0,
):
    """This operator generates a onehot Tensor from input Tensor.

    If input Tensor's rank is `N`, the corresponding onehot Tensor's rank is `N+1`.

    Flow.one_hot is aligned with tf.one_hot operator. If you want to use torch version, you can turn on_value is set to 1, off_value is set to 0.

    Args:
        input (Tensor): The input Tensor.
        num_classes (int): The length of onehot Tensor.
        on_value (Union[int, float], optional): The fill value when `x[i] == i`. Defaults to 1.
        off_value (Union[int, float], optional): The fill value when `x[i] != i`. Defaults to 0.

    Note:

        The data type of input blob should be `int32` or `int64`.

    Returns:
        oneflow.Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input=flow.tensor(np.array([0, 3, 1, 2]).astype(np.int32), dtype=flow.int64)
        >>> out = flow.nn.functional.one_hot(input, num_classes=5)
        >>> out
        tensor([[1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0]], dtype=oneflow.int64)

    """
    if input.is_consistent:
        raise ValueError(
            "A consistent tensor can not be applied to onehot, and use tensor.to_local() to convert it to local tensor first."
        )

    if num_classes == -1:
        if input.is_lazy:
            raise ValueError(
                "The parameter num_classes must be specified when one_hot using in nn.Graph."
            )

        num_classes = input.max().numpy().item() + 1

    return flow._C.one_hot(input, num_classes, on_value, off_value)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
