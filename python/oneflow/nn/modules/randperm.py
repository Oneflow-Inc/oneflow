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
from oneflow.nn.module import Module
from oneflow import Tensor
from typing import Union


class Randperm(Module):
    def __init__(
        self,
        N: flow.int32,
        generator: flow.Generator = None,
        dtype: flow.dtype = flow.int32,
        layout=None,
        device: Union[str, flow.device] = "cpu",
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        if generator is None:
            generator = flow.Generator()
        if layout is not None:
            print(
                "WARNING:",
                "oneflow.randperm.layout",
                "will not be used. Layout is not  supported yet.",
            )
        if pin_memory:
            print(
                "WARNING:",
                "pin_memory",
                "will not be used. pin_memory is not supported yet.",
            ),
        assert N > 0

        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.pin_memory = pin_memory
        self.generator = generator
        self.N = N

    def forward(self, out=None):
        res = flow.F.randperm(self.N, self.generator)
        res = res.to(self.device, self.dtype)
        res.requires_grad = self.requires_grad
        return res


def randperm(
    N: flow.int32,
    generator=None,
    out: flow.Tensor = None,
    dtype=flow.int64,
    layout=None,
    device: flow.device = flow.device("cpu"),
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor:
    r"""
    Returns a random permutation of integers from ``0`` to ``n - 1``.

    Args:
        n (int): the upper bound (exclusive)
    
    Keyword args:
        {generator(:class:`oneflow.Generator`, optional)}:  a pseudorandom number generator for sampling
        out (Tensor): output Tensor,not supported yet.
        dtype (:class:`oneflow.dtype`, optional): the desired data type of returned tensor.
            Default: ``oneflow.int64``.
        layout: layout is not supported yet.
        device: the desired device of returned tensor. Default: cpu.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: False.
        pin_memory(bool, optional):pin_memory is not supported yet.

    Example::
    .. code-block:: python

        >>> import oneflow as flow
        >>> generator = flow.Generator()
        >>> generator.manual_seed(0)
        >>> flow.randperm(5, generator=generator)
        tensor([2, 4, 3, 0, 1], dtype=oneflow.int64)
    """
    return Randperm(N, generator, dtype, layout, device, requires_grad, pin_memory)(out)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
