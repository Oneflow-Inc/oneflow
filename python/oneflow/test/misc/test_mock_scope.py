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
        with test_case.assertRaises(Exception) as context:
            from torch import noexist
        test_case.assertTrue(
            "oneflow.noexist is not implemented" in str(context.exception)
        )
        mock.disable()
        with test_case.assertRaises(Exception) as context:
            from torch import noexist
        test_case.assertTrue(
            "cannot import name 'noexist' from 'torch'" in str(context.exception)
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
            from test_mock_simple import f

            test_case.assertEqual(f(), "oneflow")


if __name__ == "__main__":
    unittest.main()
