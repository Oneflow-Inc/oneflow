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

from oneflow.framework.tensor import Tensor
import oneflow as flow
from typing import Union, List


def depend(input: Tensor, depend: Union[Tensor, List[Tensor]]) -> Tensor:
    r"""
    Add control dependency to guarantee OP A is executed before OP B.
    Used to prevent OPs from being rearranged or eliminated during graph compilation.

    Args:
        input (Tensor): a tensor intended to input OP B
        depend (Tensor or List[Tensor]): one of the output tensors of OP A (support passing in multiple tensors form different OP)

    Returns:
        Tensor: the identity of "input" tensor

    Examples:
        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        >>> import oneflow.nn.functional as F
        >>> class Model(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.OP_A = nn.Linear(128, 128)
        ...         self.OP_B = nn.Linear(128, 128)
        ...
        ...     def forward(self, x):
        ...         x1 = self.OP_A(x)
        ...         x = F.depend(x, x1)
        ...         return self.OP_B(x)
        ...
        >>> model = Model()
        >>> class Graph(nn.Graph):
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.model = model
        ...
        ...     def build(self, x):
        ...         return self.model(x)
        ...
        >>> graph = Graph()
        >>> x = flow.randn([1, 128], dtype=flow.float32)
        >>> y = graph(x)
    """
    # avoid performance loss in eager mode
    if not input.is_lazy:
        return input

    # avoid self-loop
    if isinstance(depend, Tensor) and input is depend:
        raise RuntimeError('"input" and "depend" can NOT be the same tensor.')

    if isinstance(depend, List):
        for idx, t_depend in enumerate(depend):
            if input is t_depend:
                raise RuntimeError(
                    '"input" and "depend[%d]" are the same tensor, which is not allowed.'
                    % idx
                )

    return flow._C.depend(input, depend)
