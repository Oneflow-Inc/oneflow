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
from typing import Sequence, Tuple

import numpy as np

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.ops.array_ops import (
    GetSliceAttrs,
    check_slice_tup_list,
)


class Slice(Module):
    def __init__(
        self, start: Tuple[int, ...], stop: Tuple[int, ...], step: Tuple[int, ...]
    ) -> None:
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, x):
        return flow.F.slice(x, start=self.start, stop=self.stop, step=self.step)


class SliceUpdate(Module):
    def __init__(
        self, start: Tuple[int, ...], stop: Tuple[int, ...], step: Tuple[int, ...]
    ) -> None:
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, x, update):
        return flow.F.slice_update(
            x, update, start=self.start, stop=self.stop, step=self.step
        )


class LogicalSliceAssign(Module):
    def __init__(
        self, start: Tuple[int, ...], stop: Tuple[int, ...], step: Tuple[int, ...]
    ) -> None:
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, x, update):
        if update.dtype != x.dtype:
            update = update.to(dtype=x.dtype)
        return flow.F.logical_slice_assign(
            x, update, start=self.start, stop=self.stop, step=self.step
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
