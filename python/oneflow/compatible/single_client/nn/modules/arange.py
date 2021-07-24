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


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
