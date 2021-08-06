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
from typing import Optional, Union
import oneflow as flow


class Randint(flow.nn.Module):
    def __init__(
        self,
        low: flow.int64,
        high: flow.int64,
        size: tuple,
        generator: flow.Generator = None,
        dtype: flow.dtype = flow.int64,
        layout=None,
        device=None,
        placement=None,
        sbp=None,
        requires_grad=False,
    ) -> None:
        super().__init__()

        if generator is None:
            generator = flow.Generator()
        if layout is not None:
            print(
                "WARNING:",
                "oneflow.randperm.layout",
                "will not be used. Layout is not supported yet.",
            )

        if isinstance(device, str):
            device = flow.device(device)
        self.device = device
        if placement is None:
            if device is None:
                self.device = flow.device("cpu")
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

        self.dtype = dtype
        self.requires_grad = requires_grad
        assert low < high
        self.generator = generator
        self.low = low
        self.high = high
        self.size = size

    def forward(self):
        if self.placement is not None:
            res = flow.F.consistent_randint(
                self.low, self.high, self.size,self.placement, self.sbp, self.generator
            )
        else:
            res = flow.F.randint(self.low, self.high,self.size, self.device, self.generator)
        res.requires_grad = self.requires_grad
        return res


def randint(
    low: flow.int64 = 0,
    high: Union[int, tuple] = None,
    size: tuple = None,
    generator: flow.Generator = None,
    dtype: flow.dtype = flow.int64,
    layout=None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: flow._oneflow_internal.sbp.sbp = None,
    requires_grad: bool = False
) -> flow.Tensor:
    r"""Returns a tensor filled with random integers generated uniformly from  :math:`[ \text{low},\text{high} )`.
    

    The shape of the tensor is defined by the variable argument size.

    Args:
        low (int, optional):Lowest integer to be drawn from the distribution. Default: 0.

        high (int):One above the highest integer to be drawn from the distribution.

        size (tuple):a tuple defining the shape of the output tensor.
   
    Keyword args:
        generator(:class:`oneflow.Generator`, optional):  a pseudorandom number generator for sampling
        dtype (:class:`oneflow.dtype`, optional): the desired data type of returned tensor.
            Default: ``oneflow.int64``.
        layout: layout is not supported yet.
        device: the desired device of returned tensor. Default: cpu.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: False.

    Returns:
        oneflow.Tensor: The result Tensor of given size.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> generator = flow.Generator()
        >>> generator.manual_seed(0)
        >>> flow.randint(10,(1,10),generator=generator)
        tensor([[5, 5, 7, 8, 6, 8, 5, 8, 4, 6]], dtype=oneflow.int64)
    """
    if type(high) is tuple:
        size = high
        low, high = 0, low
    return Randint(low, high, size, generator, dtype, layout, device, placement,sbp,requires_grad)()


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
