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

import oneflow as flow
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op


class Arange(Module):
    def __init__(
        self,
        start: int = 0,
        end: int = None,
        step: int = 1,
        dtype: flow.dtype = None,
        device: Union[str, flow.device] = "cpu",
        requires_grad: bool = False,
    ) -> None:
        super().__init__()
        assert end > start, "end should be larger than start"
        assert step <= end - start, "step is ilegal"

        self.start = start
        self.end = end
        self.step = step
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad

        self._op_arange = (
            flow.builtin_op("range").Output("out").Attr("dtype", flow.int64).Build()
        )

    def forward(self):
        tmp = self._op_arange(start=self.start, delta=self.step, limit=self.end)[0]
        tmp.requires_grad = self.requires_grad

        if isinstance(self.device, str):
            device = flow.device(self.device)
        else:
            device = self.device

        res = tmp.to(device, dtype=self.dtype)
        return res


@oneflow_export("arange")
@experimental_api
def arange_op(
    start: int = 0,
    end: int = None,
    step: int = 1,
    dtype: flow.dtype = flow.int64,
    device: Union[str, flow.device] = "cpu",
    requires_grad: bool = False,
):
    r"""
    Returns a 1-D tensor of size :math:`\left\lfloor \frac{\text{end} - \text{start}}{\text{step}} \right\rfloor + 1`
    with values from :attr:`start` to :attr:`end` with step :attr:`step`. Step is
    the gap between two values in the tensor.

    .. math::
        \text{out}_{i+1} = \text{out}_i + \text{step}.

    Args:
        start (float): the starting value for the set of points. Default: ``0``.
        end (float): the ending value for the set of points
        step (float): the gap between each pair of adjacent points. Default: ``1``.

    Keyword args:
    dtype: If `dtype` is not given, the `dtype` is inferred to be the default dtype.

    For example: 

    .. code-block:: python 

        import oneflow.experimental as flow
        y = flow.arange(0, 5)
        # [0, 1, 2, 3, 4]

    """
    if end is None:
        end = start
        start = 0
    return Arange(start, end, step, dtype, device, requires_grad)()
