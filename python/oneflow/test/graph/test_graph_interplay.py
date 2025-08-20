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

# Must import torch before oneflow, otherwise torch.jit.trace will raise error:
#  terminate called after throwing an instance of 'pybind11::stop_iteration'
import torch
import oneflow.unittest


def _test_relu(test_case, device, from_script=False):
    from typing import List
    import torch
    from oneflow.utils.backend.torch_compile import register_ofrt

    input_arr = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
            [0.8901075, -0.49976737, -1.07153746],
            [-0.44872912, -1.07275683, 0.06256855],
            [-0.22556897, 0.74798368, 0.90416439],
            [0.48339456, -2.32742195, -0.59321527],
        ],
        dtype=np.float32,
    )
    x = torch.tensor(input_arr, device=device)
    eager_out = torch.relu(x)

    os.environ["ofrt_from_script"] = str(from_script)
    os.environ["ofrt_enable_graph"] = "1"

    @torch.compile(backend="ofrt")
    def fn(x):
        y = torch.relu(x)
        return y

    compile_out = fn(x)
    test_case.assertTrue(
        np.allclose(
            compile_out.cpu().detach().numpy(),
            eager_out.cpu().detach().numpy(),
            1e-05,
            1e-05,
        )
    )
    compile_out = fn(x)
    test_case.assertTrue(
        np.allclose(
            compile_out.cpu().detach().numpy(),
            eager_out.cpu().detach().numpy(),
            1e-05,
            1e-05,
        )
    )


def _test_linear(test_case, device):
    from typing import List
    import torch
    from oneflow.utils.backend.torch_compile import register_ofrt

    os.environ["ofrt_from_script"] = "0"
    os.environ["ofrt_enable_graph"] = "1"

    linear = torch.nn.Linear(3, 8, False)
    linear = linear.to(device)
    input_arr = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
            [0.8901075, -0.49976737, -1.07153746],
            [-0.44872912, -1.07275683, 0.06256855],
            [-0.22556897, 0.74798368, 0.90416439],
            [0.48339456, -2.32742195, -0.59321527],
        ],
        dtype=np.float32,
    )
    x = torch.tensor(input_arr, device=device)
    torch.nn.init.constant_(linear.weight, 2.3)
    eager_out = linear(x)

    @torch.compile(backend="ofrt")
    def fn(x):
        y = linear(x)
        return y

    compile_out = fn(x)
    test_case.assertTrue(
        np.allclose(
            compile_out.cpu().detach().numpy(),
            eager_out.cpu().detach().numpy(),
            1e-05,
            1e-05,
        )
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@oneflow.unittest.skip_unless_1n1d()
class TestAsTorchBackend(oneflow.unittest.TestCase):
    def _test_relu_with_fx(test_case):
        _test_relu(test_case, "cuda", False)

    def _test_relu_with_script(test_case):
        _test_relu(test_case, "cuda", True)

    def test_linear_with_fx(test_case):
        _test_linear(test_case, "cuda")


if __name__ == "__main__":
    unittest.main()
