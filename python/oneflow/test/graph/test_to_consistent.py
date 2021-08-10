import unittest
import os
import numpy as np

import oneflow as flow
import oneflow.unittest


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


class MyModule1(flow.nn.Module):
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
            y = y.to_consistent(sbp=flow.sbp.broadcast)
            # print(f"post y shape: {y.shape}, placement: {y.placement}, sbp: {y.sbp}")
        return self.activation(y)


class MyModule2(flow.nn.Module):
    def __init__(self, weight):
        assert isinstance(weight, flow._oneflow_internal.Tensor)
        super().__init__()
        self.weight = flow.nn.Parameter(weight)
        self.activation = flow.nn.ReLU()

    def forward(self, x):
        print(f"weight shape: {self.weight.shape}, placement: {self.weight.placement}, sbp: {self.weight.sbp}")
        if self.weight.is_consistent:
            y = self.weight.to_consistent(grad_sbp=flow.sbp.broadcast)
        z = flow.F.matmul(y, x, transpose_b=True)
        return self.activation(z)


class MyGraph(flow.nn.Graph):
    def __init__(self, module, optimizer=None):
        super().__init__()
        self.module = module
        if optimizer is not None:
            self.add_optimizer("sgd", optimizer)

    def build(self, x):
        y = self.module(x)
        if self.config.training:
            y.backward()
        return y


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class ToConsistentGraphTestCase(oneflow.unittest.TestCase):
    def test_fwd_P2B(test_case):
        rank = flow.distributed.get_rank()
        # pid = os.getpid()
        # print(f"[{pid}][{rank}] ToConsistentGraphTestCase.test_fwd_P2B")

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

        # print(f"c_x shape: {c_x.shape}, placement: {c_x.placement}, sbp: {c_x.sbp}")
        # print(f"c_y shape: {c_y.shape}, placement: {c_y.placement}, sbp: {c_y.sbp}")

        m = MyModule1(c_y)
        g = MyGraph(m)

        g_z = g(c_x)
        # print(f"g_z shape: {g_z.shape}, placement: {g_z.placement}, sbp: {g_z.sbp}")

        test_case.assertTrue(np.allclose(z.numpy(), g_z.to_local().numpy()))

    def test_bwd_P2B(test_case):
        rank = flow.distributed.get_rank()
        # pid = os.getpid()
        # print(f"[{pid}][{rank}] ToConsistentGraphTestCase.test_bwd_P2B")

        local_x = flow.Tensor(x, dtype=flow.float32, device=flow.device(f"cuda:{rank}"))
        local_y = flow.Tensor(y, dtype=flow.float32, device=flow.device(f"cuda:{rank}"))

        z = flow.F.matmul(
            local_y,
            flow.cat([local_x, local_x], dim=0),
            transpose_b=True,
        )
        z = flow.F.relu(z)

        placement = flow.placement("cuda", {0: [0, 1]})
        c_x = local_x.to_consistent(placement=placement, sbp=flow.sbp.split(0))
        c_y = local_y.to_consistent(placement=placement, sbp=flow.sbp.broadcast)

        m = MyModule2(c_y)
        optimizer = flow.optim.SGD(m.parameters(), lr=1.0)
        g = MyGraph(m, optimizer)

        g_z = g(c_x)
        # print(f"g_z shape: {g_z.shape}, placement: {g_z.placement}, sbp: {g_z.sbp}")
        test_case.assertTrue(g_z.is_consistent)
        test_case.assertTrue(g_z.sbp[0] == flow.sbp.split(1))
        # S(1) -> B not supported yet
        # c_z = g_z.to_consistent(sbp=flow.sbp.broadcast)
        c_z = g_z.transpose(0, 1).to_consistent(sbp=flow.sbp.broadcast)
        # print(f"c_z shape: {c_z.shape}, placement: {c_z.placement}, sbp: {c_z.sbp}")
        test_case.assertTrue(np.allclose(z.numpy().T, c_z.to_local().numpy()))

        import time

        time.sleep(5)

        print(m.weight.to_local().numpy())

        e_y = c_y.detach()
        print(f"e_y shape: {e_y.shape}, placement: {e_y.placement}, sbp: {e_y.sbp}")
        e_m = MyModule2(e_y)
        e_z = e_m(c_x)
        print(f"e_z shape: {e_z.shape}, placement: {e_z.placement}, sbp: {e_z.sbp}")
        print(e_y.to_local().numpy())


    # def test_multi_graph(test_case):
    #     rank = flow.distributed.get_rank()
    #     # pid = os.getpid()
    #     # print(f"[{pid}][{rank}] ToConsistentGraphTestCase.test_multi_graph")

    #     local_x = flow.Tensor(x, dtype=flow.float32, device=flow.device(f"cuda:{rank}"))
    #     local_y = flow.Tensor(y, dtype=flow.float32, device=flow.device(f"cuda:{rank}"))

    #     placement = flow.placement("cuda", {0: [0, 1]})
    #     c_x = local_x.to_consistent(placement=placement, sbp=flow.sbp.split(0))
    #     c_y = local_y.to_consistent(placement=placement, sbp=flow.sbp.broadcast)


if __name__ == "__main__":
    unittest.main()
