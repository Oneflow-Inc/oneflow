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

# Eager FFTs on one thread while new training graphs, each captured into CUDA graphs on its first
# step, run on another. Every FFT call makes a new plan, so plans are made while kernels are being
# captured on the graph's actor threads.
_SCRIPT = textwrap.dedent(
    """
    import threading

    import numpy as np
    import oneflow as flow

    rng = np.random.RandomState(0)


    class TrainGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = flow.nn.Sequential(*[flow.nn.Linear(64, 64) for _ in range(32)]).to("cuda")
            self.add_optimizer(flow.optim.SGD(self.model.parameters(), lr=0.01))

        def build(self, x):
            loss = self.model(x).sum()
            loss.backward()
            return loss


    x = flow.tensor(rng.uniform(-1, 1, (8, 64)), dtype=flow.float32, device="cuda")
    z_np = rng.uniform(-1, 1, (4, 1024)).astype(np.float32)
    z = flow.tensor(z_np, device="cuda")
    expected = np.fft.rfft(z_np, axis=-1)
    done = threading.Event()
    errors = []


    def run_ffts():
        try:
            while not done.is_set():
                y = flow.fft.rfft(z, dim=-1)
                np.testing.assert_allclose(y.numpy(), expected, rtol=1e-4, atol=1e-4)
        except Exception as e:
            errors.append(e)


    fft_thread = threading.Thread(target=run_ffts)
    fft_thread.start()
    try:
        for _ in range(8):
            assert np.isfinite(TrainGraph()(x).numpy())
    finally:
        done.set()
        fft_thread.join()
    if errors:
        raise errors[0]
    """
)


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGraphCudaGraphFftPlan(flow.unittest.TestCase):
    def test_eager_fft_plan_during_cuda_graph_capture(test_case):
        env = dict(os.environ, ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH="1")
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
