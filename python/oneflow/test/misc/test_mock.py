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
from oneflow.mock_torch import mock

class TestMock(flow.unittest.TestCase):
    def test_simple_mock(test_case):
        import torch
        test_case.assertTrue(torch.__package__ == 'oneflow')
    def test_submod(test_case):
        import torch.nn
        test_case.assertTrue(torch.nn.__package__ == 'oneflow.nn')
        torch.nn.Graph
    def test_file(test_case):
        import torch.version
        test_case.assertTrue(torch.version.__version__ == flow.__version__)
    def test_from(test_case):
        from torch import nn
        from torch.version import __version__
        test_case.assertTrue(nn.__package__ == 'oneflow.nn')
        test_case.assertTrue(__version__ == flow.__version__)
    def test_error(test_case):
        with test_case.assertRaises(Exception) as context:
            from torch import noexist
        test_case.assertTrue(
            "oneflow.noexist is not implemented"
            in str(context.exception)
        )
if __name__ == "__main__":
    mock()
    unittest.main()
