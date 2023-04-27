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

import oneflow.unittest


def _test_relu(test_case, device):
    from typing import List
    import torch
    from torch._dynamo.backends.registry import register_backend
    from torch._dynamo.backends.common import fake_tensor_unsupported
    from torch import fx

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

    def to_of_transform(
        gm: torch.fx.GraphModule, tracer_class: type = fx.Tracer
    ) -> torch.fx.GraphModule:
        for node in gm.graph.nodes:
            # Checks if we're calling a function (i.e:
            # torch.add)
            if node.op == "call_function":
                # The target attribute is the function
                # that call_function calls.
                if node.target == torch.relu:
                    node.target = oneflow.relu

        gm.graph.lint()
        gm.recompile()
        return gm

    @register_backend
    @fake_tensor_unsupported
    def oneflowc(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        # TODO(): fxGraphModule to nn.Graph
        print("my_compiler() called with FX graph:")
        gm.graph.print_tabular()
        print("gm ", gm)

        import oneflow as flow

        of_gm = to_of_transform(gm)

        @flow.nn.Graph.trace
        def oneflow_fn(inputs):
            # with flow.mock_torch.enable(lazy=True):
            #     import torch
            #     outs = torch.relu(inputs)
            outs = of_gm.forward(inputs)
            return outs

        oneflow_fn.debug(1)

        def from_to_torch(inputs):
            flow_inputs = flow.utils.tensor.from_torch(inputs)
            flow_outs = oneflow_fn(flow_inputs)
            # TODO(): general output process
            outs = flow.utils.tensor.to_torch(flow_outs[0])
            return (outs,)

        return from_to_torch

    @torch.compile(backend="oneflowc")
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
    from torch._dynamo.backends.registry import register_backend
    from torch._dynamo.backends.common import fake_tensor_unsupported

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

    def get_of():
        # TODO(): transform torch fx code to oneflow code
        import oneflow as flow

        linear = flow.nn.Linear(3, 8, False)
        linear = linear.to(device)
        flow.nn.init.constant_(linear.weight, 2.3)

        class LinearGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.my_linear = linear

            def build(self, x):
                return self.my_linear(x)

        linear_g = LinearGraph()
        linear_g.debug(1)
        return linear_g

    g = None

    def torch_interplay(x):
        import oneflow as flow

        x = flow.utils.tensor.from_torch(x)
        nonlocal g
        if g is None:
            g = get_of()
        # TODO(): This is a special pack trick, try to make it general.
        return (flow.utils.tensor.to_torch(g(x)),)

    @register_backend
    @fake_tensor_unsupported
    def oneflowc(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        print("my_compiler() called with FX graph:")
        gm.graph.print_tabular()
        gm.forward = torch_interplay
        return gm.forward  # return a python callable

    @torch.compile(backend="oneflowc")
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
    def test_relu(test_case):
        _test_relu(test_case, "cuda")

    def test_linear(test_case):
        _test_linear(test_case, "cuda")


if __name__ == "__main__":
    unittest.main()
