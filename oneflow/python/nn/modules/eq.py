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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op


class Eq(Module):
    def __init__(self) -> None:
        super().__init__()
        self.eq_op = (
            flow.builtin_op("broadcast_equal").Input("x").Input("y").Output("z").Build()
        )

    def forward(self, input, other):
        if isinstance(other, flow.Tensor):
            for i in range(len(input.size())):
                assert (
                    input.shape[i] >= other.shape[i]
                ), "The second tensor's shape should broadcastable with the first argument."
        elif isinstance(other, int) or isinstance(other, float):
            raise NotImplementedError(
                "Unsupport data type, int or float data type are not support yet!"
            )
        else:
            raise NotImplementedError(
                "Unsupport data type, The second argument can be a tensor whose shape is broadcastable with the first argument."
            )

        return self.eq_op(input, other)[0]


@oneflow_export("eq", "equal")
@register_tensor_op("eq")
@experimental_api
def eq_op(input, other):
    r"""
    Computes element-wise equality.
    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.

    Args:
    input (Tensor): the tensor to compare
    other (Tensor): the tensor to compare

    Returns:
    A boolean tensor that is True where :attr:`input` is equal to :attr:`other` and False elsewhere

    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        import numpy as np

        input = flow.Tensor(np.array([2, 3, 4, 5]), dtype=flow.float32)
        other = flow.Tensor(np.array([2, 3, 4, 1]), dtype=flow.float32)

        y = flow.eq(input, other)
        # [1 1 1 0]

    """
    return Eq()(input, other)
