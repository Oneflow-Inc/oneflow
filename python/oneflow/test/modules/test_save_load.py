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
import warnings
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest


class CustomModuleForSaveLoad(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = flow.nn.Parameter(flow.randn(1, 3, 3, 3))

    def forward(self, x):
        return self.param + x


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestSaveLoad(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_load_map_location(test_case):
        x = flow.ones(1, 2, 3)
        y = flow.ones(2, 3, 4)
        with tempfile.NamedTemporaryFile() as f:
            flow.save({"x": x, "y": y}, f.name)
            loaded = flow.load(f.name, map_location="cuda")
        assert np.array_equal(loaded["x"].numpy(), x.numpy())
        assert loaded["x"].device == flow.device("cuda")
        assert np.array_equal(loaded["y"].numpy(), y.numpy())
        assert loaded["y"].device == flow.device("cuda")

        with tempfile.NamedTemporaryFile() as f:
            flow.save({"x": x, "y": y}, f.name)
            loaded = flow.load(f.name, map_location="cpu")
        assert np.array_equal(loaded["x"].numpy(), x.numpy())
        assert loaded["x"].device == flow.device("cpu")
        assert np.array_equal(loaded["y"].numpy(), y.numpy())
        assert loaded["y"].device == flow.device("cpu")

        x = x.to_global(sbp=flow.sbp.broadcast, placement=flow.placement("cuda", [0]))
        y = y.to_global(sbp=flow.sbp.broadcast, placement=flow.placement("cuda", [0]))

        with tempfile.NamedTemporaryFile() as f:
            flow.save({"x": x, "y": y}, f.name, global_dst_rank=0)
            loaded = flow.load(
                f.name, global_src_rank=0, map_location=flow.placement("cuda", [0])
            )
        assert np.array_equal(loaded["x"].numpy(), x.numpy())
        assert loaded["x"].placement == flow.placement("cuda", [0])
        assert np.array_equal(loaded["y"].numpy(), y.numpy())
        assert loaded["y"].placement == flow.placement("cuda", [0])

        with tempfile.NamedTemporaryFile() as f:
            flow.save({"x": x, "y": y}, f.name, global_dst_rank=0)
            loaded = flow.load(
                f.name, global_src_rank=0, map_location=flow.placement("cpu", [0])
            )
        assert np.array_equal(loaded["x"].numpy(), x.numpy())
        assert loaded["y"].placement == flow.placement("cpu", [0])
        assert np.array_equal(loaded["y"].numpy(), y.numpy())
        assert loaded["y"].placement == flow.placement("cpu", [0])

    @flow.unittest.skip_unless_1n1d()
    def test_save_dir(test_case):
        m1 = CustomModuleForSaveLoad()
        with tempfile.TemporaryDirectory() as save_dir:
            flow.save(m1.state_dict(), save_dir, save_as_external_data=True)
            loaded_state_dict = flow.load(save_dir)
        m2 = CustomModuleForSaveLoad()
        m2.load_state_dict(loaded_state_dict)
        test_case.assertTrue(np.array_equal(m1.param.numpy(), m2.param.numpy()))

    @flow.unittest.skip_unless_1n1d()
    def test_save_dir_fault_tolerance(test_case):
        m1 = CustomModuleForSaveLoad()
        with tempfile.TemporaryDirectory() as save_dir:
            flow.save(m1.state_dict(), save_dir, save_as_external_data=True)
            with open(os.path.join(save_dir, "random_file"), "w") as fp:
                fp.write("nothing")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded_state_dict = flow.load(save_dir)
        m2 = CustomModuleForSaveLoad()
        m2.load_state_dict(loaded_state_dict)
        test_case.assertTrue(np.array_equal(m1.param.numpy(), m2.param.numpy()))

    @flow.unittest.skip_unless_1n1d()
    def test_save_state_dict(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.param1 = flow.nn.Parameter(flow.Tensor(32, 1024, 1024))
                self.param2 = flow.nn.Parameter(flow.Tensor(32, 1024, 1024))

            def forward(self):
                return self.param1 + self.param2

        m = CustomModule()
        res1 = m()
        state_dict = m.state_dict()
        with tempfile.NamedTemporaryFile() as f:
            flow.save(state_dict, f.name)
            test_case.assertTrue(os.path.exists(f.name))
            loaded_state_dict = flow.load(f.name)
            m.load_state_dict(loaded_state_dict)
        res2 = m()
        test_case.assertTrue(np.array_equal(res1.numpy(), res2.numpy()))

    def _test_save_and_load_global_from_nested_dict(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = flow.nn.Parameter(flow.randn(3, 32, 3, 3))

            def forward(self):
                return self.param

        m1 = CustomModule()
        m1 = m1.to_global(
            flow.placement("cuda", range(1, 3)), flow.sbp.broadcast
        ).to_global(sbp=flow.sbp.split(1))
        m2 = CustomModule()
        m2 = m2.to_global(flow.placement("cuda", range(1, 3)), flow.sbp.broadcast)
        res1 = m1() + m2()
        state_dict1 = m1.state_dict()
        state_dict2 = m2.state_dict()
        state_dict = {"m1": state_dict1, "m2": state_dict2}

        with tempfile.TemporaryDirectory() as dir:
            filename = os.path.join(dir, "tmp")
            with test_case.assertRaises(Exception):
                flow.save(state_dict, filename)

            global_src_dst_rank = 0
            flow.save(state_dict, filename, global_dst_rank=global_src_dst_rank)
            rank = flow.env.get_rank()
            if rank != global_src_dst_rank:
                test_case.assertFalse(os.path.exists(filename))

            m1 = CustomModule()
            m1 = m1.to_global(
                flow.placement("cuda", [[0, 1], [2, 3]]),
                [flow.sbp.broadcast, flow.sbp.broadcast],
            ).to_global(sbp=[flow.sbp.split(1), flow.sbp.broadcast])
            m2 = CustomModule()
            m2 = m2.to_global(
                flow.placement("cuda", [[0, 1], [2, 3]]),
                [flow.sbp.broadcast, flow.sbp.broadcast],
            ).to_global(sbp=[flow.sbp.broadcast, flow.sbp.split(1)])

            with test_case.assertRaises(Exception):
                loaded_state_dict = flow.load(filename)
                m1.load_state_dict(loaded_state_dict["m1"])

            loaded_state_dict = flow.load(filename, global_src_rank=global_src_dst_rank)
            test_case.assertEqual(len(loaded_state_dict), 2)
            m1.load_state_dict(loaded_state_dict["m1"])
            m2.load_state_dict(loaded_state_dict["m2"])
            res2 = m1() + m2()

        test_case.assertTrue(np.array_equal(res1.numpy(), res2.numpy()))

    @flow.unittest.skip_unless_1n4d()
    def test_save_and_load_global_from_nested_dict_1n4d(test_case):
        test_case._test_save_and_load_global_from_nested_dict()

    @flow.unittest.skip_unless_2n2d()
    def test_save_and_load_global_from_nested_dict_2n2d(test_case):
        test_case._test_save_and_load_global_from_nested_dict()

    @flow.unittest.skip_unless_1n1d()
    def test_load_pytorch_weights(test_case):
        for device in ["cpu", "cuda"]:
            for map_location in [None, flow.device("cuda:0")]:
                conv_torch = torch.nn.Conv2d(3, 3, 3).to(device)

                conv_flow1 = flow.nn.Conv2d(3, 3, 3).to(device)
                with tempfile.NamedTemporaryFile() as f:
                    torch.save(conv_torch.state_dict(), f.name)
                    conv_flow1.load_state_dict(
                        flow.load(f.name, map_location=map_location)
                    )
                test_case.assertTrue(
                    np.array_equal(
                        conv_torch.weight.detach().cpu().numpy(),
                        conv_flow1.weight.numpy(),
                    )
                )

                conv_flow2 = flow.nn.Conv2d(3, 3, 3).to(device)
                with tempfile.NamedTemporaryFile() as f:
                    torch.save({"weights": conv_torch.state_dict()}, f.name)
                    conv_flow2.load_state_dict(
                        flow.load(f.name, map_location=map_location)["weights"]
                    )
                test_case.assertTrue(
                    np.array_equal(
                        conv_torch.weight.detach().cpu().numpy(),
                        conv_flow2.weight.numpy(),
                    )
                )

    @flow.unittest.skip_unless_1n2d()
    def test_load_pytorch_weights_global(test_case):
        for device in ["cpu", "cuda"]:
            for map_location in [None, flow.placement.all("cuda")]:
                conv_torch = torch.nn.Conv2d(3, 3, 3).to(device)

                all_placement = flow.placement.all(device)
                conv_flow1 = flow.nn.Conv2d(3, 3, 3).to_global(
                    all_placement, flow.sbp.broadcast
                )
                with tempfile.NamedTemporaryFile() as f:
                    if flow.env.get_rank() == 0:
                        torch.save(conv_torch.state_dict(), f.name)
                    conv_flow1.load_state_dict(
                        flow.load(f.name, map_location=map_location, global_src_rank=0)
                    )
                if flow.env.get_rank() == 0:
                    test_case.assertTrue(
                        np.array_equal(
                            conv_torch.weight.detach().cpu().numpy(),
                            conv_flow1.weight.numpy(),
                        )
                    )

                conv_flow2 = flow.nn.Conv2d(3, 3, 3).to_global(
                    all_placement, flow.sbp.broadcast
                )
                with tempfile.NamedTemporaryFile() as f:
                    if flow.env.get_rank() == 0:
                        torch.save({"weights": conv_torch.state_dict()}, f.name)
                    conv_flow2.load_state_dict(
                        flow.load(f.name, map_location=map_location, global_src_rank=0)[
                            "weights"
                        ]
                    )
                if flow.env.get_rank() == 0:
                    test_case.assertTrue(
                        np.array_equal(
                            conv_torch.weight.detach().cpu().numpy(),
                            conv_flow2.weight.numpy(),
                        )
                    )

    @flow.unittest.skip_unless_1n1d()
    def test_save_load_module_directly(test_case):
        x = flow.randn(1, 3, 3, 3)

        m = CustomModuleForSaveLoad()

        with tempfile.NamedTemporaryFile() as f:
            flow.save(m, f.name)
            new_m = flow.load(f.name)
            res = m(x)
            new_res = new_m(x)
            test_case.assertTrue(np.array_equal(res.numpy(), new_res.numpy()))

        m = flow.nn.parallel.DistributedDataParallel(m)
        test_case.assertTrue(m._is_ddp_module)

        with tempfile.NamedTemporaryFile() as f:
            flow.save(m, f.name)
            new_m = flow.load(f.name)
            test_case.assertTrue(new_m._is_ddp_module)
            res = m(x)
            new_res = new_m(x)
            test_case.assertTrue(np.array_equal(res.numpy(), new_res.numpy()))

    def test_load_old_dir_data(test_case):
        test_data_dir = Path(__file__).parent / "save_load_test_data"
        m1 = nn.Conv2d(3, 3, 3)
        params = flow.load(test_data_dir / "3x3_i3o3_conv2d_params")
        m1.load_state_dict(params)

        m2 = flow.load(test_data_dir / "3x3_i3o3_conv2d")

        x = flow.randn(1, 3, 3, 3)
        y1 = m1(x)
        y2 = m2(x)
        test_case.assertTrue(np.array_equal(y1.numpy(), y2.numpy()))

    def test_pytorch_non_tensor(test_case):
        with tempfile.NamedTemporaryFile() as f:
            torch.save({"a": 2}, f.name)
            res = flow.load(f.name, map_location="cpu")
        test_case.assertTrue(isinstance(res, dict))
        test_case.assertEqual(len(res), 1)
        test_case.assertEqual(res["a"], 2)


if __name__ == "__main__":
    unittest.main()
