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
        y = flow._C.matmul(x, self.weight, transpose_b=True)
        # print(f"y shape: {y.shape}, placement: {y.placement}, sbp: {y.sbp}")
        if y.is_global:
            y = y.to_global(sbp=flow.sbp.broadcast)
            # print(f"post y shape: {y.shape}, placement: {y.placement}, sbp: {y.sbp}")
        return self.activation(y)


class MyModule2(flow.nn.Module):
    def __init__(self, weight):
        assert isinstance(weight, flow._oneflow_internal.Tensor)
        super().__init__()
        self.weight = flow.nn.Parameter(weight)
        self.activation = flow.nn.ReLU()

    def forward(self, x):
        # print(f"weight shape: {self.weight.shape}, placement: {self.weight.placement}, sbp: {self.weight.sbp}")
        if self.weight.is_global:
            y = self.weight.to_global(grad_sbp=flow.sbp.broadcast)
        z = flow._C.matmul(y, x, transpose_b=True)
        out = self.activation(z).sum()
        if self.weight.is_global:
            out = out.to_global(sbp=flow.sbp.broadcast)
        return out


class MyModule3(flow.nn.Module):
    def __init__(self, transpose_a=False, transpose_b=False):
        super().__init__()
        self.activation = flow.nn.ReLU()
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def forward(self, x, y):
        z = flow._C.matmul(x, y, self.transpose_a, self.transpose_b)
        if z.is_global:
            z = z.to_global(sbp=flow.sbp.broadcast)
        return self.activation(z)


class GlobalToModule(flow.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x.to(self.device)


class FreeTensorModule(flow.nn.Module):
    def __init__(self, shape, placement, sbp):
        super().__init__()
        self.shape = shape
        self.placement = placement
        self.sbp = sbp

    def forward(self, x):
        y = flow.ones(
            self.shape, dtype=flow.float32, placement=self.placement, sbp=self.sbp
        )
        return flow._C.matmul(x, y, transpose_b=True)


class ToPlacementModule(flow.nn.Module):
    def __init__(self, placement):
        super().__init__()
        self.placement = placement

    def forward(self, x):
        return x.to_global(placement=self.placement)


class MyGraph(flow.nn.Graph):
    def __init__(self, module, optimizer=None):
        super().__init__()
        self.module = module
        if optimizer is not None:
            self.add_optimizer(optimizer)

    def build(self, *arg):
        y = self.module(*arg)
        if self.config.training:
            y.backward()
        return y


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class ToGlobalGraphTestCase(oneflow.unittest.TestCase):
    # @unittest.skipIf(True, "")
    def test_fwd_P2B(test_case):
        """ compare eager fwd and lazy bwd
        """
        rank = flow.env.get_rank()
        # pid = os.getpid()
        # print(f"[{pid}][{rank}] ToGlobalGraphTestCase.test_fwd_P2B")

        local_x = flow.tensor(x, dtype=flow.float32, device=flow.device(f"cuda:{rank}"))
        local_y = flow.tensor(y, dtype=flow.float32, device=flow.device(f"cuda:{rank}"))

        z = flow._C.matmul(
            flow.cat([local_x, local_x], dim=1),
            flow.cat([local_y, local_y], dim=1),
            transpose_b=True,
        )
        z = flow._C.relu(z)
        # print(f"z shape: {z.shape}, device: {z.device}")
        # print(z.numpy())

        placement = flow.placement("cuda", ranks=[0, 1])
        sbp = flow.sbp.split(1)
        c_x = local_x.to_global(placement=placement, sbp=sbp)
        c_y = local_y.to_global(placement=placement, sbp=sbp)

        # print(f"c_x shape: {c_x.shape}, placement: {c_x.placement}, sbp: {c_x.sbp}")
        # print(f"c_y shape: {c_y.shape}, placement: {c_y.placement}, sbp: {c_y.sbp}")

        m = MyModule1(c_y)
        g = MyGraph(m)

        g_z = g(c_x)
        # print(f"g_z shape: {g_z.shape}, placement: {g_z.placement}, sbp: {g_z.sbp}")
        # print(g_z.to_local().numpy())
        test_case.assertTrue(np.allclose(z.numpy(), g_z.to_local().numpy()))

    # @unittest.skipIf(True, "")
    def test_bwd_P2B(test_case):
        """ compare eager bwd and lazy bwd
        """
        rank = flow.env.get_rank()
        # pid = os.getpid()
        # print(f"[{pid}][{rank}] ToGlobalGraphTestCase.test_bwd_P2B")

        local_x = flow.tensor(x, dtype=flow.float32, device=flow.device(f"cuda:{rank}"))
        local_y = flow.tensor(y, dtype=flow.float32, device=flow.device(f"cuda:{rank}"))

        z = flow._C.matmul(
            local_y, flow.cat([local_x, local_x], dim=0), transpose_b=True,
        )
        z = flow._C.relu(z)
        z = z.sum()

        placement = flow.placement("cuda", ranks=[0, 1])
        c_x = local_x.to_global(placement=placement, sbp=flow.sbp.split(0))
        c_y = local_y.to_global(placement=placement, sbp=flow.sbp.broadcast)

        m = MyModule2(c_y)
        optimizer = flow.optim.SGD(m.parameters(), lr=1.0)
        g = MyGraph(m, optimizer)

        g_z = g(c_x)
        # print(f"g_z shape: {g_z.shape}, placement: {g_z.placement}, sbp: {g_z.sbp}")
        test_case.assertTrue(g_z.is_global)
        test_case.assertTrue(g_z.sbp[0] == flow.sbp.broadcast)
        # S(1) -> B not supported yet
        # c_z = g_z.to_global(sbp=flow.sbp.broadcast)
        # print(f"c_z shape: {c_z.shape}, placement: {c_z.placement}, sbp: {c_z.sbp}")
        test_case.assertTrue(np.allclose(z.numpy(), g_z.to_local().numpy()))

        e_y = c_y.detach()
        # print(f"e_y shape: {e_y.shape}, placement: {e_y.placement}, sbp: {e_y.sbp}")
        e_m = MyModule2(e_y)
        e_z = e_m(c_x)
        # print(f"e_z shape: {e_z.shape}, placement: {e_z.placement}, sbp: {e_z.sbp}")
        e_z.backward()

        test_case.assertTrue(
            np.allclose(c_y.to_local().numpy(), e_y.to_local().numpy())
        )

    # @unittest.skipIf(True, "")
    def test_multi_graph(test_case):
        """ compare two lazy fwd
        """
        rank = flow.env.get_rank()
        # pid = os.getpid()
        # print(f"[{pid}][{rank}] ToGlobalGraphTestCase.test_multi_graph")

        local_x = flow.tensor(x, dtype=flow.float32, device=flow.device(f"cuda:{rank}"))
        local_y = flow.tensor(y, dtype=flow.float32, device=flow.device(f"cuda:{rank}"))

        placement = flow.placement("cuda", ranks=[0, 1])
        x1 = local_x.to_global(placement=placement, sbp=flow.sbp.broadcast)
        y1 = local_y.to_global(placement=placement, sbp=flow.sbp.broadcast)
        # B * B -> B -> B
        m1 = MyModule3(transpose_b=True)
        g1 = MyGraph(m1)

        slice_obj = slice(
            int(rank * local_x.shape[0] / 2), int((rank + 1) * local_x.shape[0] / 2)
        )
        x2 = local_x[slice_obj, :]
        x2 = x2.to_global(placement=placement, sbp=flow.sbp.split(0))
        y2 = local_y.to_global(placement=placement, sbp=flow.sbp.broadcast)
        # S(0) * B -> S(0) -> B
        m2 = MyModule3(transpose_b=True)
        g2 = MyGraph(m2)

        x3 = local_x[
            :, int(rank * local_x.shape[1] / 2) : int((rank + 1) * local_x.shape[1] / 2)
        ]
        x3 = x3.to_global(placement=placement, sbp=flow.sbp.split(1))
        y3 = local_y[
            :, int(rank * local_y.shape[1] / 2) : int((rank + 1) * local_y.shape[1] / 2)
        ]
        y3 = y3.to_global(placement=placement, sbp=flow.sbp.split(1))
        # S(1) * S(0) -> P -> B
        m3 = MyModule3(transpose_b=True)
        g3 = MyGraph(m3)

        z1 = g1(x1, y1)
        # print(f"z1 shape: {z1.shape}, placement: {z1.placement}, sbp: {z1.sbp}")
        # print(z1.to_local().numpy())
        z2 = g2(x2, y2)
        # print(f"z2 shape: {z2.shape}, placement: {z2.placement}, sbp: {z2.sbp}")
        # print(z2.to_local().numpy())
        z3 = g3(x3, y3)
        # print(f"z3 shape: {z3.shape}, placement: {z3.placement}, sbp: {z3.sbp}")
        # print(z3.to_local().numpy())

        test_case.assertTrue(np.allclose(z1.to_local().numpy(), z2.to_local().numpy()))
        test_case.assertTrue(np.allclose(z1.to_local().numpy(), z3.to_local().numpy()))

    # @unittest.skipIf(True, "")
    def test_global_to(test_case):
        c_x = flow.ones(
            (4, 3), placement=flow.placement("cpu", ranks=[0, 1]), sbp=flow.sbp.split(0)
        )

        global_to = GlobalToModule("cuda")
        g_global_to = MyGraph(global_to)

        e = global_to(c_x)
        test_case.assertTrue(e.is_cuda)
        test_case.assertTrue(e.is_global)
        test_case.assertTrue(e.sbp[0] == flow.sbp.split(0))

        g = g_global_to(c_x)
        test_case.assertTrue(g.is_cuda)
        test_case.assertTrue(g.is_global)
        test_case.assertTrue(g.sbp[0] == flow.sbp.split(0))

        test_case.assertTrue(np.allclose(e.to_local().numpy(), g.to_local().numpy()))

    # @unittest.skipIf(True, "")
    def test_free_tensor_to_global(test_case):
        local_x = flow.tensor(x, dtype=flow.float32, device="cpu")
        placement = flow.placement("cuda", ranks=[0, 1])
        c_x = local_x.to_global(placement, flow.sbp.split(0))

        m = FreeTensorModule((3, 10), placement, flow.sbp.broadcast)
        g = MyGraph(m)

        eager_out = m(c_x)
        test_case.assertTrue(eager_out.is_cuda)
        test_case.assertTrue(eager_out.is_global)
        test_case.assertTrue(eager_out.sbp[0] == flow.sbp.split(0))

        graph_out = g(c_x)
        test_case.assertTrue(graph_out.is_cuda)
        test_case.assertTrue(graph_out.is_global)
        test_case.assertTrue(graph_out.sbp[0] == flow.sbp.split(0))

        test_case.assertTrue(
            np.allclose(eager_out.to_local().numpy(), graph_out.to_local().numpy())
        )

    # @unittest.skipIf(True, "")
    def test_to_placement(test_case):
        rank = flow.env.get_rank()
        # pid = os.getpid()
        # print(f"[{pid}][{rank}] ToGlobalGraphTestCase.test_to_placement")

        if rank == 0:
            x = flow.ones((2, 3), dtype=flow.float32)
        elif rank == 1:
            x = flow.empty(tuple())
        else:
            raise ValueError

        c_x = x.to_global(
            placement=flow.placement("cpu", ranks=[0]), sbp=flow.sbp.broadcast
        )
        # print(f"c_x shape: {c_x.shape}, placement: {c_x.placement}, sbp: {c_x.sbp}")

        p1 = flow.placement("cpu", ranks=[0, 1])
        m1 = ToPlacementModule(p1)
        g1 = MyGraph(m1)
        y1 = g1(c_x)

        # print(f"y1 shape: {y1.shape}, placement: {y1.placement}, sbp: {y1.sbp}")
        test_case.assertTrue(y1.placement == p1)
        test_case.assertTrue(y1.sbp[0] == flow.sbp.broadcast)
        test_case.assertTrue(y1.to_local().numpy().mean() == 1.0)

        p2 = flow.placement("cuda", ranks=[0, 1])
        m2 = ToPlacementModule(p2)
        g2 = MyGraph(m2)
        y2 = g2(y1)

        # print(f"y2 shape: {y2.shape}, placement: {y2.placement}, sbp: {y2.sbp}")
        test_case.assertTrue(y2.placement == p2)
        test_case.assertTrue(y2.sbp[0] == flow.sbp.broadcast)
        test_case.assertTrue(y2.to_local().numpy().mean() == 1.0)

    # @unittest.skipIf(True, "")
    def test_to_dtype(test_case):
        x = flow.ones((2, 3), dtype=flow.int32, device="cpu")

        placement = flow.placement("cpu", ranks=[0, 1])
        c_x = flow.ones(
            (2, 3), dtype=flow.int32, placement=placement, sbp=flow.sbp.broadcast
        )

        class CastModule(flow.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                return x.to(dtype=self.dtype)

        m = CastModule(flow.float32)
        g = MyGraph(m)

        e_x = m(x)
        e_c_x = m(c_x)
        # NOTE(chengcheng):
        #   There are two BUG in this test script:
        #   1. first call and second call input tensor meta is NOT same
        #   2. nn.Graph NOT support local input with multi-rank yet.
        # g_x = g(x)
        g_c_x = g(c_x)

        test_case.assertTrue(e_x.dtype == flow.float32)
        # test_case.assertTrue(g_x.dtype == flow.float32)
        test_case.assertTrue(e_c_x.dtype == flow.float32)
        test_case.assertTrue(g_c_x.dtype == flow.float32)


class MyModule5(flow.nn.Module):
    def __init__(self, transpose_a=False, transpose_b=False, sbp=[]):
        super().__init__()
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.sbp = sbp

    def forward(self, x, y):
        z = flow._C.matmul(x, y, self.transpose_a, self.transpose_b)
        assert z.is_global
        assert len(z.sbp) == len(self.sbp)
        return z.to_global(sbp=self.sbp)


@unittest.skipIf(True, "")
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n4d()
class ToGlobal2DGraphTestCase(oneflow.unittest.TestCase):
    def test_matmul(test_case):
        placement = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
        x = flow.ones(
            (4, 6), placement=placement, sbp=[flow.sbp.split(0), flow.sbp.split(1)]
        )
        y = flow.ones(
            (4, 6), placement=placement, sbp=[flow.sbp.broadcast, flow.sbp.split(1)]
        )
        z = flow._C.matmul(x, y, transpose_b=True)
        print(f"z shape: {z.shape}, placement: {z.placement}, sbp: {z.sbp}")

        # m = MyModule5(transpose_b=True, sbp=[flow.sbp.split(0), flow.sbp.broadcast])
        # z = m(x, y)
        # print(f"z shape: {z.shape}, placement: {z.placement}, sbp: {z.sbp}")


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestLazy1dTo2dGlobal(flow.unittest.TestCase):
    def test_lazy_1d_to_2d_sbp(test_case):
        P_1d = flow.placement(
            device_type="cuda", device_ids={0: range(4)}, hierarchy=(4,)
        )
        P_2d = flow.placement(
            device_type="cuda", device_ids={0: range(4)}, hierarchy=(2, 2)
        )
        B = flow.sbp.broadcast

        class Test1dTo2dModule(flow.nn.Module):
            def forward(self, x):
                return x.to_global(placement=P_2d, sbp=[B, B])

        class Test1dTo2dGraph(flow.nn.Graph):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def build(self, x):
                return self.model(x)

        class Test2dTo1dModule(flow.nn.Module):
            def forward(self, x):
                return x.to_global(placement=P_1d, sbp=[B])

        class Test2dTo1dGraph(flow.nn.Graph):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def build(self, x):
                return self.model(x)

        model_1d_to_2d = Test1dTo2dModule()
        graph_1d_to_2d = Test1dTo2dGraph(model_1d_to_2d)

        x = flow.zeros(4, 4, 4, 4, sbp=[B, B], placement=P_2d)
        x = x.to_global(placement=P_1d, sbp=[B])
        test_case.assertTrue(x.sbp == (B,))
        test_case.assertTrue(x.placement == P_1d)
        y = graph_1d_to_2d(x)
        test_case.assertTrue(y.sbp == (B, B))
        test_case.assertTrue(y.placement == P_2d)

        model_2d_to_1d = Test2dTo1dModule()
        graph_2d_to_1d = Test2dTo1dGraph(model_2d_to_1d)
        z = graph_2d_to_1d(y)
        test_case.assertTrue(z.sbp == x.sbp)
        test_case.assertTrue(z.placement == x.placement)


if __name__ == "__main__":
    unittest.main()
