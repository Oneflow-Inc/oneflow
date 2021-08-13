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
        n,
        generator: flow.Generator = None,
        dtype: flow.dtype = flow.int32,
        layout=None,
        device: Union[flow.device, str, None] = None,
        placement: flow.placement = None,
        sbp: flow._oneflow_internal.sbp.sbp = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        if generator is None:
            generator = flow.default_generator()
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
        assert n > 0
        if isinstance(device, str):
            device = flow.device(device)
        if placement is None:
            if device is None:
                device = flow.device("cpu")
        else:
            assert device is None

        self.placement = placement
        self.sbp = sbp
        if placement is not None:
            assert isinstance(sbp, (flow.sbp.sbp, tuple, list)), "sbp: %s" % sbp
            if isinstance(self.sbp, flow.sbp.sbp):
                self.sbp = (self.sbp,)
            else:
                for elem in sbp:
                    assert isinstance(elem, flow.sbp.sbp), "sbp: %s" % sbp
            assert len(self.sbp) == len(placement.hierarchy)
        else:
            assert sbp is None, "sbp: %s" % sbp

        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.pin_memory = pin_memory
        self.generator = generator
        self.n = n

    def forward(self, out=None):
        if self.placement is not None:
            res = flow.F.consistent_randperm(
                self.n, self.placement, self.sbp, self.generator
            )
        else:
            res = flow.F.randperm(self.n, self.device, self.generator)
        res.requires_grad = self.requires_grad
        return res.to(dtype=self.dtype)


def randperm(
    n: flow.int32,
    generator: flow.Generator = None,
    out: Tensor = None,
    dtype: flow.dtype = flow.int32,
    layout=None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: flow._oneflow_internal.sbp.sbp = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor:
    r"""
    Returns a random permutation of integers from ``0`` to ``n - 1``.

    Args:
        n (int): the upper bound (exclusive)
    
    Keyword args:
        generator(:class:`oneflow.Generator`, optional):  a pseudorandom number generator for sampling
        out (Tensor, optional): output Tensor,not supported yet.
        dtype (:class:`oneflow.dtype`, optional): the desired data type of returned tensor.
            Default: ``oneflow.int32``.
        layout: layout is not supported yet.
        device: the desired device of returned tensor. Default: cpu.
        placement:(:class:`flow.placement`, optional): The desired device of returned consistent tensor. If None,
            will construct local tensor.
        sbp: (:class:`flow.sbp`, optional): The desired sbp of returned consistent tensor. It must be equal with the
            numbers of placement.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: False.
        pin_memory(bool, optional):pin_memory is not supported yet.

    Example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> generator = flow.Generator()
        >>> generator.manual_seed(0)
        >>> flow.randperm(5, generator=generator)
        tensor([2, 4, 3, 0, 1], dtype=oneflow.int32)
    """
    return Randperm(
        n, generator, dtype, layout, device, placement, sbp, requires_grad, pin_memory
    )(out)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
