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


class Log1p(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("log1p").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@register_tensor_op("log1p")
def log1p_op(input):
    """Returns a new tensor with the natural logarithm of (1 + input).

    .. math::
        \\text{out}_{i}=\\log_e(1+\\text{input}_{i})

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> x = flow.Tensor(np.array([1.3, 1.5, 2.7]))
        >>> out = flow.log1p(x).numpy()
        >>> out
        array([0.8329091 , 0.91629076, 1.3083328 ], dtype=float32)

    """
    return Log1p()(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
