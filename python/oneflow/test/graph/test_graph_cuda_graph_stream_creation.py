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

# A training graph captured into CUDA graphs, followed at once by the first device-to-host copy in
# eager mode. The copy creates a stream, with its BLAS and solver handles, while the graph may still
# be capturing its first step on another thread.
_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import oneflow as flow

    rng = np.random.RandomState(0)
    model = flow.nn.Sequential(*[flow.nn.Linear(64, 64) for _ in range(32)]).to("cuda")


    class TrainGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model
            self.add_optimizer(flow.optim.SGD(model.parameters(), lr=0.01))

        def build(self, x):
            loss = self.model(x).sum()
            loss.backward()
            return loss


    train_graph = TrainGraph()
    x = flow.tensor(rng.uniform(-1, 1, (8, 64)), dtype=flow.float32, device="cuda")
    y_np = rng.uniform(-1, 1, (8, 64)).astype(np.float32)
    y = flow.tensor(y_np, device="cuda")
    train_graph(x)
    np.testing.assert_array_equal(y.cpu().numpy(), y_np)
    for _ in range(3):
        assert np.isfinite(train_graph(x).numpy())
    """
)


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGraphCudaGraphStreamCreation(flow.unittest.TestCase):
    def test_eager_stream_creation_during_cuda_graph_capture(test_case):
        env = dict(
            os.environ,
            ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH="1",
            # On ROCm, makes rocBLAS and hipSOLVER handles create a hipBLASLt handle, as they do by
            # default on some GPUs.
            ROCBLAS_USE_HIPBLASLT="1",
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
