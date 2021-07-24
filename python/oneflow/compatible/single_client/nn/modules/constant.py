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

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.common_types import _size_any_t
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.nn.modules.utils import _single


class _ConstantBase(Module):
    def __init__(
        self,
        size: Union[_size_any_t, flow.Size],
        value: Union[float, int],
        dtype: Optional[flow.dtype],
        device: Union[flow.device, str] = None,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()
        assert size is not None, "shape must not be None!"
        assert isinstance(
            size, (int, tuple, flow.Size)
        ), "shape should be int or tuple int!"
        self.device = device
        self.requires_grad = requires_grad
        size = _single(size)
        if dtype is None:
            dtype = flow.float32
        if device is None:
            self.device = flow.device("cpu")
        self.shape = size
        self.value = value
        self.dtype = dtype

    def forward(self):
        res = flow.F.constant(self.shape, self.value, self.dtype)
        res = res.to(device=self.device)
        res.requires_grad = self.requires_grad
        return res


class Ones(_ConstantBase):
    def __init__(self, size, dtype=None, device=None, requires_grad=False):
        super().__init__(size, 1, dtype, device, requires_grad)


class Zeros(_ConstantBase):
    def __init__(self, size, dtype=None, device=None, requires_grad=False):
        super().__init__(size, 0, dtype, device, requires_grad)


class ZerosLike(Module):
    def __init__(self):
        super().__init__()

    def forward(self, other):
        return flow.F.zeros_like(other)


class OnesLike(Module):
    def __init__(self):
        super().__init__()

    def forward(self, other):
        return flow.F.ones_like(other)


class NewOnes(Module):
    def __init__(
        self,
        size: Union[_size_any_t, flow.Size] = None,
        dtype: Optional[flow.dtype] = None,
        device: Union[flow.device, str] = None,
        requires_grad: bool = False,
    ):
        super().__init__()
        self.device = device
        self.requires_grad = requires_grad
        if size != None:
            size = _single(size)
        self.size = size
        self.dtype = dtype

    def forward(self, x):
        new_size = self.size
        new_dtype = self.dtype
        new_device = self.device
        new_requires_grad = self.requires_grad
        if self.size is None:
            new_size = x.shape
        if self.dtype is None:
            new_dtype = x.dtype
        if self.device is None:
            new_device = x.device
        assert isinstance(
            new_size, (int, tuple, flow.Size)
        ), f"size parameter not correct, please check!"
        assert isinstance(
            new_dtype, flow.dtype
        ), f"dtype parameter not correct, please check!"
        assert isinstance(
            new_device, (str, flow.device)
        ), f"device parameter not correct, please check!"
        assert isinstance(
            new_requires_grad, bool
        ), f"requires_grad parameter not correct, please check!"
        res = flow.F.constant(new_size, 1.0, new_dtype)
        res = res.to(new_device)
        res.requires_grad = new_requires_grad
        return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
