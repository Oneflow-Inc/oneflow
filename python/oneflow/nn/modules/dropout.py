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
import random
import sys

import oneflow as flow
import oneflow.framework.id_util as id_util
from oneflow.nn.modules.module import Module


class _DropoutNd(Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, but got {}".format(p)
            )
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return "p={}, inplace={}".format(self.p, self.inplace)


class Dropout(_DropoutNd):
    def __init__(self, p: float = 0.5, inplace: bool = False, generator=None):
        _DropoutNd.__init__(self, p, inplace)
        self.p = p
        self.generator = generator

    def forward(self, x, addend=None):
        return flow._C.dropout(
            x,
            self.p,
            self.training,
            self.inplace,
            self.generator,
            addend=addend if addend is not None else None,
        )


class Dropout1d(Dropout):
    def forward(self, x, addend=None):
        return flow._C.dropout1d(x, self.p, self.training)


class Dropout2d(Dropout):
    def forward(self, x, addend=None):
        return flow._C.dropout2d(x, self.p, self.training)


class Dropout3d(Dropout):
    def forward(self, x, addend=None):
        return flow._C.dropout3d(x, self.p, self.training)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
