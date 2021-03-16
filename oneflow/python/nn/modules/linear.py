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
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import Tensor
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("nn.Identity")
class Identity(Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = flow.nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = flow.Tensor(np.random.rand(2, 3, 4, 5))
        >>> output = m(input)

    """

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input
