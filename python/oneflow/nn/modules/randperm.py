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

class Randperm(Module):
    def __init__(self,N:int, generator=None, dtype=flow.int32, layout=None, device=None, requires_grad=False,
                 pin_memory=False) -> None:
        super().__init__()

        

        if generator is None:
            generator = flow.Generator()
        if layout is not None:
            print(
                "WARNING:",
                "oneflow.randperm.layout",
                "will not be used. Layout is not currently supported."
            )
        if pin_memory is not None:
            print(
                "WARNING:",
                "pin_memory",
                "will not be used. pin_memory is not currently supported."
            ),
        if isinstance(device, str):
            device = flow.device(device)
        else:
            device = device if device is not None else flow.device("cpu")
        
        assert isinstance(device, flow.device)
        assert isinstance(dtype, flow.dtype)
        # assert isinstance(requires_grad, bool)
        # assert isinstance(pin_memory, bool)
        assert N>0

        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.pin_memory = pin_memory
        self.generator = generator
        self.N=N
    def forward(self, out=None):
        return flow.F.randperm(self.N,self.dtype,self.generator)


def randperm(n, generator=None, out=None, dtype=flow.int64, layout=None, device=None, requires_grad=False,
             pin_memory=False) -> Tensor:
    r"""
    Returns a random permutation of integers from ``0`` to ``n - 1``.
    Args:
        n (int): the upper bound (exclusive)
    Keyword args:
        {generator}: custom generator is not currently supported.
        out (Tensor): output Tensor.
        dtype (:class:`oneflow.dtype`, optional): the desired data type of returned tensor.
            Default: ``oneflow.int64``.
        {layout}: layout is not currently supported.
        {device}
        {requires_grad}
        {pin_memory}
    Example::
    .. code-block:: python
        >>> torch.randperm(4)
        tensor([2, 1, 0, 3])
    """
    return Randperm(generator, dtype, layout, device, requires_grad, pin_memory)(out)

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
