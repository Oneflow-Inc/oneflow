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


class Lgamma(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("lgamma").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("lgamma") #flow.lgamma
@register_tensor_op("lgamma") #t.lgamma
@experimental_api
def lgamma_op(x):
    """This operator computes the :math:`Gamma(x)` value.

    The equation is:

    .. math::

        out = \int_{0}^{\infty}t^{x-1}*e^{-t}\mathrm{d}{t}

    Args:
        x (oneflow._oneflow_internal.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def lgamma_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.lgamma(x)


        x = np.array([1.3, 1.5, 2.7]).astype(np.float32)
        out = lgamma_Job(x)

        # out [-0.1081748  -0.12078223  0.4348206 ]

    """
    return Lgamma()(x)
