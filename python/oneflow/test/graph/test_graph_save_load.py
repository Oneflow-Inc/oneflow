import unittest
import os
import numpy as np

import oneflow as flow
import oneflow.unittest


def _test_linear_graph_save_load(test_case, device):
    def train_with_graph():
        linear = flow.nn.Linear(3, 8)
        linear = linear.to(device)
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 0.23)
        of_sgd = flow.optim.SGD(linear.parameters(), lr=0.001, momentum=0.9)

        x = flow.tensor(
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
            dtype=flow.float32,
            device=device,
            requires_grad=False,
        )

        class LinearTrainGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.add_optimizer(of_sgd)

            def build(self, x):
                out = self.linear(x)
                out = out.sum()
                out.backward()
                return out

        linear_t_g = LinearTrainGraph()
        of_graph_out = linear_t_g(x)
        state_dict = linear_t_g.state_dict()
        print(state_dict)

    train_with_graph()


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestLinearGraphSaveLoad(oneflow.unittest.TestCase):
    def test_linear_graph_save_load_gpu(test_case):
        _test_linear_graph_save_load(test_case, flow.device("cuda"))

    def test_linear_graph_save_load_cpu(test_case):
        _test_linear_graph_save_load(test_case, flow.device("cpu"))


if __name__ == "__main__":
    unittest.main()
