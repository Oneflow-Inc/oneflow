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
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


class CombinedMarginLoss(Module):
    def __init__(self, m1: float = 1, m2: float = 0, m3: float = 0) -> None:
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def forward(self, x, label):
        depth = x.shape[1]
        (y, theta) = flow.F.combined_margin_loss(x, label,
            m1 = self.m1, m2 = self.m2, m3 = self.m3, depth = depth)
        return y


def combined_margin_loss_op(x, label, m1: float = 1, m2: float = 0, m3: float = 0):
    """The operation implement "loss_name == 'margin_softmax'" in insightface.
    insightface's margin_softmax loss implement by several operators, we combined them for speed up.

    Args:
        x (oneflow.Tensor): A Tensor
        label (oneflow.Tensor): label with integer data type

    Returns:
        oneflow.Tensor: A Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_x = np.random.uniform(low=-1, high=1, size=(10, 6)).astype(np.float32)
        >>> np_label = np.random.randint(0, 6, size=(10)).astype(np.int32)
        >>> x = flow.Tensor(x, dtype=flow.float32)
        >>> label = flow.Tensor(label, dtype=flow.int32)   
        >>> out = flow.combined_margin_loss(x, label, 0.3, 0.5, 0.4)

    """
    return CombinedMarginLoss(m1, m2, m3)(x, label)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
