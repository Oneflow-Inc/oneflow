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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class Repeat(Module):
    def __init__(self, sizes) -> None:
        super().__init__()
        self.sizes = sizes

    def forward(self, input):
        repeat = self.sizes
        for repeat_v in repeat:
            assert repeat_v > 0
        input_shape = input.shape
        assert len(repeat) >= len(input_shape)
        in_reshape = []
        out_reshape = []
        expand_dim = []
        diff = len(repeat) - len(input_shape)
        for i in range(len(repeat) - 1, -1, -1):
            if i >= diff:
                if repeat[i] > 1:
                    if input_shape[i - diff] > 1:
                        in_reshape.insert(0, input_shape[i - diff])
                        in_reshape.insert(0, 1)
                        expand_dim.insert(0, input_shape[i - diff])
                        expand_dim.insert(0, repeat[i])
                        out_reshape.insert(0, input_shape[i - diff] * repeat[i])
                    else:
                        in_reshape.insert(0, input_shape[i - diff])
                        expand_dim.insert(0, repeat[i])
                        out_reshape.insert(0, repeat[i])
                else:
                    in_reshape.insert(0, input_shape[i - diff])
                    expand_dim.insert(0, input_shape[i - diff])
                    out_reshape.insert(0, input_shape[i - diff])
            else:
                expand_dim.insert(0, repeat[i])
                out_reshape.insert(0, repeat[i])
        new_tensor = flow.experimental.reshape(input, in_reshape)
        tmp_tensor = new_tensor.expand(*expand_dim)
        out = flow.experimental.reshape(tmp_tensor, out_reshape)
        return out


@register_tensor_op("repeat")
def repeat_op(x, sizes):
    """This operator repeat the input tensor to a larger size along the specified dimensions.

    Args:
        x (oneflow.compatible.single_client.Tensor): The input Tensor.
        size (Sequence[int]): The number of times to repeat this tensor along each dimension

    Returns:
        oneflow.compatible.single_client.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> x = np.array([[[[0, 1]],
        ...               [[2, 3]],
        ...               [[4, 5]]]]).astype(np.int32)

        >>> input = flow.Tensor(x)
        >>> out = input.repeat(sizes=(1, 1, 2, 2))
        >>> out.shape
        flow.Size([1, 3, 2, 4])
    """
    return Repeat(sizes=sizes)(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
