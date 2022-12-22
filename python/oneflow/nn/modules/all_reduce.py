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
from oneflow.nn.modules.module import Module

from typing import Sequence


class AllReduce(Module):
    def __init__(self, parallel_conf_str: str):
        super().__init__()
        self._op = (
            flow.stateful_op("eager_ccl_all_reduce").Input("in").Output("out").Build()
        )
        self.parallel_conf = parallel_conf_str

    def forward(self, x):
        assert x.device.type == "cuda"
        assert x.device.index == flow.env.get_local_rank()
        return flow._C.dispatch_eager_ccl_all_reduce(self._op, parallel_conf)
