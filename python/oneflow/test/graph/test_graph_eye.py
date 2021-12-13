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
import numpy as np
import random
import oneflow as flow
import oneflow.unittest
from test_util import generate_graph


@flow.unittest.skip_unless_1n1d()
class TestEyeGraph(oneflow.unittest.TestCase):
    def test_eye_graph(test_case):
        n = random.randint(1, 10)
        m = random.randint(1, 10)

        eye_fn = lambda: flow.eye(n, m)
        y_eager = eye_fn()
        eye_graph = generate_graph(eye_fn)
        y_lazy = eye_graph()
        test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))


if __name__ == "__main__":
    unittest.main()
