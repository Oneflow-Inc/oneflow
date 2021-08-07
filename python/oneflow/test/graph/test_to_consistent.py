import unittest
import os
import numpy as np

import oneflow as flow
import oneflow.unittest


class MyModule(flow.nn.Module):
    def __init__(self, weight):
        assert isinstance(weight, flow._oneflow_internal.Tensor)
        super().__init__()
        self.weight = flow.nn.Parameter(weight)
        self.activation = flow.nn.ReLU()

    def forward(self, x):
        # print(f"x shape: {x.shape}, placement: {x.placement}, sbp: {x.sbp}")
        # print(
        #     f"weight shape: {self.weight.shape}, placement: {self.weight.placement}, sbp: {self.weight.sbp}"
        # )
        y = flow.F.matmul(x, self.weight, transpose_b=True)
        # print(f"y shape: {y.shape}, placement: {y.placement}, sbp: {y.sbp}")
        if y.is_consistent:
            y = y.to_consistent(placement=self.weight.placement, sbp=flow.sbp.broadcast)
            # print(f"post y shape: {y.shape}, placement: {y.placement}, sbp: {y.sbp}")
        return self.activation(y)


class MyGraph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def build(self, x):
        return self.module(x)


x = np.array(
    [
        [
            0.21490018,
            0.22043167,
            0.1605895,
            0.25424683,
            0.12975895,
            0.49967155,
            0.04753795,
            0.7518577,
            0.38964537,
            0.01955934,
        ],
        [
            0.16392729,
            0.41410774,
            0.05424517,
            0.7668146,
            0.08050849,
            0.5763975,
            0.42364502,
            0.4950619,
            0.9608427,
            0.11889187,
        ],
    ]
)

y = np.array(
    [
        [
            0.9903706,
            0.11213686,
            0.29525927,
            0.79380244,
            0.70357895,
            0.6950597,
            0.52552456,
            0.32304054,
            0.6997739,
            0.15671141,
        ],
        [
            0.76867193,
            0.59983397,
            0.07774717,
            0.07815815,
            0.30385414,
            0.7366552,
            0.4607681,
            0.40554753,
            0.8290172,
            0.8405671,
        ],
        [
            0.8900324,
            0.5274955,
            0.80989295,
            0.71331054,
            0.8076364,
            0.94833183,
            0.04778554,
            0.23992656,
            0.57683426,
            0.81757474,
        ],
    ]
)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class ToConsistentGraphTestCase(oneflow.unittest.TestCase):
    def test_case1(test_case):
        rank = flow.distributed.get_rank()
        print(
            f"GPTDataLoaderDistributedTestCase.test_case1 on rank {rank} {os.getpid()}"
        )

        local_x = flow.Tensor(x, dtype=flow.float32, device=flow.device(f"cuda:{rank}"))
        local_y = flow.Tensor(y, dtype=flow.float32, device=flow.device(f"cuda:{rank}"))

        z = flow.F.matmul(
            flow.cat([local_x, local_x], dim=1),
            flow.cat([local_y, local_y], dim=1),
            transpose_b=True,
        )
        z = flow.F.relu(z)

        placement = flow.placement("cuda", {0: [0, 1]})
        sbp = flow.sbp.split(1)
        c_x = local_x.to_consistent(placement=placement, sbp=sbp)
        c_y = local_y.to_consistent(placement=placement, sbp=sbp)

        print(f"c_x shape: {c_x.shape}, placement: {c_x.placement}, sbp: {c_x.sbp}")
        print(f"c_y shape: {c_y.shape}, placement: {c_y.placement}, sbp: {c_y.sbp}")

        m = MyModule(c_y)
        g = MyGraph(m)

        g_z = g(c_x)
        print(f"g_z shape: {g_z.shape}, placement: {g_z.placement}, sbp: {g_z.sbp}")

        test_case.assertTrue(np.allclose(z.numpy(), g_z.to_local().numpy()))


if __name__ == "__main__":
    unittest.main()
