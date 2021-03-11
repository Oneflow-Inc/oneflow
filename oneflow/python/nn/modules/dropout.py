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
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.id_util as id_util

@oneflow_export("nn.Dropout")
class Dropout(Module):
    def __init__(self, p=0.5, noise_shape=None, seed=None, name=None):
        super().__init__()
        self.rate = p
        self.name = name
        self.seed = seed
        self.noise_shape = noise_shape
        assert self.rate is not None and self.rate >= 0.0 and self.rate < 1.0
        if self.name is None:
            self.name = id_util.UniqueStr("Dropout_")

        self._op = (
            flow.builtin_op("dropout")
            .Name("dropout")
            .Input("in")
            .Input("mask")
            .Attr("scale", float(1.0 / (1.0 - self.rate)))
            .Output("out")
            .Build()
        )

    def forward(self, x):
        if not flow.current_global_function_desc().IsTrainable() or self.rate == 0.0:
            return x
        mask = flow.nn.random_mask_like(
            x, self.rate, self.seed, self.noise_shape, "%s-dropout_random_mask_like" % self.name
        )
        res = self._op(x, mask)[0]
        return res
