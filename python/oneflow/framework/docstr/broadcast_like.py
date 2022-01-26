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
    oneflow.broadcast_like,
    """
    This operator broadcast tensor `x` to `like_tensor` according to the broadcast_axes. 

    Args:
        x (Tensor): The input Tensor. 
        like_tensor (Tensor): The like Tensor. 
        broadcast_axes (Optional[Sequence], optional): The axes you want to broadcast. Defaults to None.

    Returns:
        [Tensor]: Broadcasted input Tensor. 

    For example: 

    .. code:: python

        >>> import oneflow as flow 

        >>> x = flow.randn(3, 1, 1)
        >>> like_tensor = flow.randn(3, 4, 5)
        >>> broadcast_tensor = flow.broadcast_like(x, like_tensor, broadcast_axes=[1, 2]) 
        >>> broadcast_tensor.shape
        oneflow.Size([3, 4, 5])
    
    """,
)
