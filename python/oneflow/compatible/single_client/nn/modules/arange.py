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

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


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

    def forward(self):
        tmp = flow.F.range(
            start=self.start, limit=self.end, delta=self.step, dtype=flow.int64
        )
        tmp.requires_grad = self.requires_grad
        if isinstance(self.device, str):
            device = flow.device(self.device)
        else:
            device = self.device
        res = tmp.to(device, dtype=self.dtype)
        return res


def arange_op(
    start: int = 0,
    end: int = None,
    step: int = 1,
    dtype: flow.dtype = flow.int64,
    device: Union[str, flow.device] = "cpu",
    requires_grad: bool = False,
):
    """
    Returns a 1-D tensor of size :math:`\\left\\lfloor \\frac{\\text{end} - \\text{start}}{\\text{step}} \\right\\rfloor + 1`
    with values from :attr:`start` to :attr:`end` with step :attr:`step`. Step is
    the gap between two values in the tensor.

    .. math::
        \\text{out}_{i+1} = \\text{out}_i + \\text{step}.

    Args:
        start (int): the starting value for the set of points. Default: ``0``.
        end (int): the ending value for the set of points
        step (int): the gap between each pair of adjacent points. Default: ``1``.

    Keyword args:
        dtype(flow.dtype, optional): If `dtype` is not given, the `dtype` is inferred to be `flow.int64`.
        device(flow.device, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: `False`.

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

        >>> y = flow.arange(0, 5)
        >>> y
        tensor([0, 1, 2, 3, 4], dtype=oneflow.int64)

    """
    if end is None:
        end = start
        start = 0
    return Arange(start, end, step, dtype, device, requires_grad)()


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
