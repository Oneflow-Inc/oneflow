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
import oneflow as flow
import oneflow.unittest
import oneflow.mock_torch as mock

"""
enable & disable mode hold a dict[str, ModuleType] like sys.modules, the keys start with 'torch'.
The two modes don't interfere with each other, sys.modules and global scope are replaced on switch.
"""


with mock.enable():
    import torch
    import torch.nn
    import torch.version
with mock.disable():
    import torch
    import torch.nn
    import torch.version


@flow.unittest.skip_unless_1n1d()
class TestMock(flow.unittest.TestCase):
    def test_with(test_case):
        with mock.enable():
            test_case.assertEqual(torch.__package__, "oneflow")
            test_case.assertEqual(torch.nn.__package__, "oneflow.nn")
            test_case.assertEqual(torch.version.__version__, flow.__version__)
        with mock.disable():
            test_case.assertEqual(torch.__package__, "torch")
            test_case.assertEqual(torch.nn.__package__, "torch.nn")
            test_case.assertEqual(torch.version.__version__, torch.__version__)

    def test_simple(test_case):
        mock.enable()
        test_case.assertEqual(torch.__package__, "oneflow")
        test_case.assertEqual(torch.nn.__package__, "oneflow.nn")
        test_case.assertEqual(torch.version.__version__, flow.__version__)

        mock.disable()

        test_case.assertEqual(torch.__package__, "torch")
        test_case.assertEqual(torch.nn.__package__, "torch.nn")
        test_case.assertEqual(torch.version.__version__, torch.__version__)

    def test_import_from(test_case):
        mock.enable()
        from torch import nn
        from torch.version import __version__

        test_case.assertEqual(nn.__package__, "oneflow.nn")
        test_case.assertEqual(__version__, flow.__version__)

        mock.disable()
        from torch import nn
        from torch.version import __version__

        test_case.assertEqual(nn.__package__, "torch.nn")
        test_case.assertEqual(__version__, torch.__version__)

    def test_error(test_case):
        mock.enable()
        with test_case.assertRaises(ImportError) as context:
            from torch import noexist
        test_case.assertTrue(
            "cannot import name 'noexist' from 'oneflow'" in str(context.exception)
        )
        with test_case.assertRaises(ModuleNotFoundError) as context:
            import torch.noexist
        test_case.assertTrue(
            "oneflow.noexist is not implemented" in str(context.exception)
        )
        mock.disable()
        with test_case.assertRaises(ImportError) as context:
            from torch import noexist
        test_case.assertTrue(
            "cannot import name 'noexist' from 'torch'" in str(context.exception)
        )
        with test_case.assertRaises(ModuleNotFoundError) as context:
            import torch.noexist
        test_case.assertTrue(
            "No module named 'torch.noexist'" in str(context.exception)
        )

    def test_nested_with(test_case):
        with mock.enable():
            test_case.assertEqual(torch.__package__, "oneflow")
            with mock.disable():
                test_case.assertEqual(torch.__package__, "torch")
            test_case.assertEqual(torch.__package__, "oneflow")
        with mock.disable():
            test_case.assertEqual(torch.__package__, "torch")
            with mock.enable():
                test_case.assertEqual(torch.__package__, "oneflow")
            test_case.assertEqual(torch.__package__, "torch")

    def test_noop_disable(test_case):
        with mock.disable():
            import torch

            test_case.assertEqual(torch.__package__, "torch")

    def test_3rd_party(test_case):
        with mock.enable():
            from mock_example import f

            test_case.assertEqual(f(), "oneflow")

    def test_env_var(test_case):
        os.environ["ONEFLOW_DISABLE_MOCK_TORCH"] = "1"

        with mock.enable():
            import torch

            test_case.assertEqual(torch.__package__, "torch")

        os.environ["ONEFLOW_DISABLE_MOCK_TORCH"] = "0"

    def test_dummy_obj_fallback(test_case):
        with mock.enable(lazy=True):
            from torch import not_exist

            test_case.assertEqual(not_exist.__name__, "oneflow.not_exist")
            x = not_exist.x
            test_case.assertEqual(x.__name__, "oneflow.not_exist.x")

    def test_mock_torchvision(test_case):
        with mock.enable(lazy=True):
            import torchvision

            model = torchvision.models.resnet18(pretrained=False)
            test_case.assertEqual(len(list(model.parameters())), 62)

    def test_mock_lazy_for_loop(test_case):
        with mock.enable(lazy=True):
            import torch

            # Test no infinite loop
            for _ in torch.not_exist:
                pass

    def test_mock_lazy_in_if(test_case):
        with mock.enable(lazy=True):
            import torch

            if torch.not_exist:
                test_case.assertTrue(False)

    def test_blacklist(test_case):
        with mock.enable(lazy=True):
            import torch
            import torch.nn.functional as F

            test_case.assertFalse(hasattr(F, "scaled_dot_product_attention"))
            test_case.assertFalse(
                hasattr(torch.nn.functional, "scaled_dot_product_attention")
            )

    def test_hazard_list(test_case):
        with mock.enable():
            import sys
            import safetensors
        test_case.assertTrue("safetensors._safetensors_rust" in sys.modules)
        import safetensors


# MUST use pytest to run this test
def test_verbose(capsys):
    with mock.enable(lazy=True, verbose=True):
        import torch.not_exist

        captured = capsys.readouterr()
        assert "oneflow.not_exist is not found in oneflow" in captured.out


if __name__ == "__main__":
    unittest.main()
