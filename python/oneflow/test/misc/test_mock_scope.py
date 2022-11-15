import unittest
import oneflow as flow
import oneflow.unittest
from oneflow.mock_torch import _enable, enable, _disable, disable, mock

import torch
import torch.nn
import torch.version

"""
Mock saves a module dict (like sys.modules) for real torch modules and oneflow modules respectively
when using enable/disable,
torch-related k-v pairs in sys.modules and global scope are replaced with the cache in `mock`
"""

import torch
import torch.nn
import torch.version


class TestMock(flow.unittest.TestCase):
    def test_with(test_case):
        with enable():
            test_case.assertTrue(torch.__package__ == "oneflow")
            test_case.assertTrue(torch.nn.__package__ == "oneflow.nn")
            test_case.assertTrue(torch.version.__version__ == flow.__version__)
        with disable():
            test_case.assertTrue(torch.__package__ == "torch")
            test_case.assertTrue(torch.nn.__package__ == "torch.nn")
            test_case.assertTrue(torch.version.__version__ == torch.__version__)

    def test_simple(test_case):
        _enable()
        test_case.assertTrue(torch.__package__ == "oneflow")
        test_case.assertTrue(torch.nn.__package__ == "oneflow.nn")
        test_case.assertTrue(torch.version.__version__ == flow.__version__)

        _disable()

        test_case.assertTrue(torch.__package__ == "torch")
        test_case.assertTrue(torch.nn.__package__ == "torch.nn")
        test_case.assertTrue(torch.version.__version__ == torch.__version__)

    def test_import_from(test_case):
        _enable()
        from torch import nn
        from torch.version import __version__

        test_case.assertTrue(nn.__package__ == "oneflow.nn")
        test_case.assertTrue(__version__ == flow.__version__)

        _disable()
        from torch import nn
        from torch.version import __version__

        test_case.assertTrue(nn.__package__ == "torch.nn")
        test_case.assertTrue(__version__ == torch.__version__)

    def test_error(test_case):
        _enable()
        with test_case.assertRaises(Exception) as context:
            from torch import noexist
        test_case.assertTrue(
            "oneflow.noexist is not implemented" in str(context.exception)
        )
        _disable()
        with test_case.assertRaises(Exception) as context:
            from torch import noexist
        test_case.assertTrue(
            "cannot import name 'noexist' from 'torch'" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
