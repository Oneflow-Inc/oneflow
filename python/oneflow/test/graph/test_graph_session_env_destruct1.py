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
import os
import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest

linear = flow.nn.Linear(3, 8, False)
input_arr = np.random.randn(8, 3).astype(np.float32)
np_weight = np.ones((3, 8)).astype(np.float32)
np_weight.fill(2.3)
x = flow.tensor(input_arr)
flow.nn.init.constant_(linear.weight, 2.3)
of_eager_out = linear(x)
np_out = np.matmul(input_arr, np_weight)
assert np.allclose(of_eager_out.numpy(), np_out, 1e-05, 1e-05)


class LinearGraphDestruct1(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.my_linear = linear

    def build(self, x):
        return self.my_linear(x)


# test graph destruction when graph is not compiled
linear_g_d_not_compiled = LinearGraphDestruct1()
print("test graph destruction when graph is not compiled")


if __name__ == "__main__":
    unittest.main()
