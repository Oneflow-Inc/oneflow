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
from typing import Optional

import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op


class One_hot(Module):
    def __init__(self, on_value=1, off_value=0, depth, dtype=None) -> None:
        super().__init__()
        self.on_value = on_value
        self.off_value = off_value
        self.depth = depth
        if dtype == None:
            dtype = flow.int32
        self.dtype = dtype

    def forward(self, x):
        return flow.F.one_hot(x, self.on_value, self.off_value, self.depth, self.dtype)




def one_hot_op(
    x, 
    depth: int, 
    on_value: Union[int, float] = 1,
    off_value: Union[int, float] = 0,
    dtype: Optional[flow.dtype] = None,
):
    """This operator generates a onehot Blob from input Blob.

    If input Blob's rank is `N`, the corresponding onehot Blob's rank is `N+1`.

    The locations represented by `x` take value `on_value`, while other locations take `off_value`.

    Args:
        x (Tensor): The input Tensor.
        depth (int): The length of onehot Blob.
        on_value (Union[int, float], optional): The fill value when `x[i] == i`. Defaults to 1.
        off_value (Union[int, float], optional): The fill value when `x[i] != i`. Defaults to 0.
        dtype (Optional[flow.dtype], optional): The output data type, it can be "oneflow.compatible.single_client.int32", "oneflow.compatible.single_client.int64", "oneflow.compatible.single_client.float", "oneflow.compatible.single_client.double". Defaults to None.

    Note:

        The data type of input blob should be `int32` or `int64`.

    Returns:
        oneflow.Tensor.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x=flow.Tensor(np.array([0, 3, 1, 2].astype(np.int32)))
        >>> out = flow.one_hot(x, depth=5)
        >>> out
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]], dtype=oneflow.float32)
    
    """
    return One_hot(on_value, off_value, depth, dtype)(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
