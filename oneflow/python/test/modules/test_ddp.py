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
import unittest
import oneflow.experimental as flow

import numpy as np


@flow.unittest.skip_unless_1n2d()
class TestAllReduce(flow.unittest.TestCase):
    def test_all_reduce(test_case):
        class Mul(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = flow.nn.Parameter(flow.Tensor([1]))

            def forward(self, x):
                return x * self.w

        local_rank = flow.distributed.get_local_rank()
        if local_rank == 0:
            x = flow.Tensor([1])
        elif local_rank == 1:
            x = flow.Tensor([2])
        else:
            raise ValueError()

        x = x.to("cuda")
        m = Mul().to("cuda")
        m = flow.ddp(m)
        y = m(x)
        y.backward()

        print(m.w.grad)


if __name__ == "__main__":
    unittest.main()
