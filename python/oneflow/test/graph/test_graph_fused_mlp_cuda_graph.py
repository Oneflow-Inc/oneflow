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
import subprocess
import sys
import textwrap
import unittest

import oneflow as flow
import oneflow.unittest

# A training graph captured into CUDA graphs, followed at once by the same step in eager mode.
# The eager backward creates the fused MLP grad kernel's BLAS handle while the graph may still be
# capturing on another thread.
_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import oneflow as flow

    k = 1024
    rng = np.random.RandomState(0)
    init = {
        "w0": rng.uniform(-1, 1, (8, k)),
        "b0": rng.uniform(-1, 1, (8,)),
        "w1": rng.uniform(-1, 1, (k, 8)),
        "b1": rng.uniform(-1, 1, (k,)),
    }
    names = list(init)


    class Mlp(flow.nn.Module):
        def __init__(self):
            super().__init__()
            for n in names:
                p = flow.nn.Parameter(flow.tensor(init[n], dtype=flow.float32, device="cuda"))
                setattr(self, n, p)

        def forward(self, x, dy):
            w0, b0, w1, b1 = [getattr(self, n).to(flow.float16) for n in names]
            out = flow._C.fused_mlp(x, [w0, w1], [b0, b1], True)
            return (out.float() * dy).sum()


    graph_model, eager_model = Mlp(), Mlp()


    class TrainGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = graph_model
            self.add_optimizer(flow.optim.SGD(graph_model.parameters(), lr=1.0))

        def build(self, x, dy):
            loss = self.model(x, dy)
            loss.backward()
            return loss


    train_graph = TrainGraph()
    for step in range(3):
        x = flow.tensor(rng.uniform(-1, 1, (2, k)), dtype=flow.float16, device="cuda")
        dy = flow.tensor(rng.uniform(-1, 1, (2, k)), dtype=flow.float32, device="cuda")
        before = {n: getattr(graph_model, n).numpy() for n in names}
        train_graph(x, dy)
        for n in names:
            getattr(eager_model, n).data = flow.tensor(before[n], device="cuda")
        eager_model(x, dy).backward()
        for n in names:
            # SGD with lr = 1 subtracts the gradient once.
            np.testing.assert_allclose(
                getattr(graph_model, n).numpy(),
                before[n] - getattr(eager_model, n).grad.numpy(),
                rtol=1e-3,
                atol=1e-3,
                err_msg="%s, step %d" % (n, step),
            )
    """
)


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGraphFusedMlpCudaGraph(flow.unittest.TestCase):
    def test_fused_mlp_async_grad_cuda_graph_with_eager(test_case):
        env = dict(
            os.environ,
            ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH="1",
            ONEFLOW_ONE_EMBEDDING_FUSED_MLP_ASYNC_GRAD="1",
        )
        result = subprocess.run(
            [sys.executable, "-c", _SCRIPT],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        test_case.assertEqual(result.returncode, 0, result.stdout.decode())


if __name__ == "__main__":
    unittest.main()
