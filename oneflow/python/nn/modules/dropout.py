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
import sys
import random
import oneflow as flow
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.id_util as id_util


@oneflow_export("nn.Dropout")
class Dropout(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self._rate = p
        self._seed = random.randint(-sys.maxsize, sys.maxsize)
        assert self._rate is not None and self._rate >= 0.0 and self._rate < 1.0
        self._scale = float(1.0 / (1.0 - self._rate))
        assert inplace == False, "Not support inplace=True yet!"
        self._op = (
            flow.builtin_op("dropout")
            .Name(self.id_util.UniqueStr("Dropout_"))
            .Input("in")
            .Input("mask")
            .Output("out")
            .Attr("scale", self._scale)
            .Build()
        )
        self._mask_op = (
            flow.builtin_op("random_mask_like")
            .Name(id_util.UniqueStr("RandomMaskLike_"))
            .Input("like")
            .Output("out")
            .Attr("rate", self._rate)
            .Attr("seed", self._seed)
            .Build()
        )

    def forward(self, x):
        if self.rate == 0.0:
            return x
        mask = self._mask_op(x)[0]
        res = self._op(x, mask)[0]
        return res
