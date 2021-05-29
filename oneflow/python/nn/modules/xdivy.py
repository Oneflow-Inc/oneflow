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


class Xdivy(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("xdivy").Input("x").Input("y").Output("z").Build()
    
    def forward(self, x, y):
        return self._op(x, y)[0]


@oneflow_export("xdivy", "xdivy")
@register_tensor_op("xdivy")
@experimental_api
def xdivy_op(x, y):
    """This operator computes the log sigmoid value of input Blob.

    Args:
        x (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        import numpy as np
        import oneflow.typing as tp

        x = flow.Tensor(
            np.array([4, 3, 5]).astype(np.float32), dtype=flow.float32
        )
        y = flow.Tensor(
            np.array([3, 2, 2]).astype(np.float32), dtype=flow.float32
        )
        out = flow.xdivy(x, y).numpy()

        # out [1.3333334 1.5       2.5      ]

    """
    return Xdivy()(x, y)
