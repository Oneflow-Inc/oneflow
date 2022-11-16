import unittest
import oneflow as flow
import oneflow.unittest
import oneflow.mock_torch as mock

"""
Mock saves a module dict (like sys.modules) for real torch modules and oneflow modules respectively
when using enable/disable,
torch-related k-v pairs in sys.modules and global scope are replaced with the cache in `mock` on switch
"""


def _import_both():
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
        _import_both()
        with mock.enable():
            test_case.assertTrue(torch.__package__ == "oneflow")
            test_case.assertTrue(torch.nn.__package__ == "oneflow.nn")
            test_case.assertTrue(torch.version.__version__ == flow.__version__)
        with mock.disable():
            test_case.assertTrue(torch.__package__ == "torch")
            test_case.assertTrue(torch.nn.__package__ == "torch.nn")
            test_case.assertTrue(torch.version.__version__ == torch.__version__)

    def test_simple(test_case):
        _import_both()
        mock.enable()
        test_case.assertTrue(torch.__package__ == "oneflow")
        test_case.assertTrue(torch.nn.__package__ == "oneflow.nn")
        test_case.assertTrue(torch.version.__version__ == flow.__version__)

        mock.disable()

        test_case.assertTrue(torch.__package__ == "torch")
        test_case.assertTrue(torch.nn.__package__ == "torch.nn")
        test_case.assertTrue(torch.version.__version__ == torch.__version__)

    def test_import_from(test_case):
        _import_both()
        mock.enable()
        from torch import nn
        from torch.version import __version__

        test_case.assertTrue(nn.__package__ == "oneflow.nn")
        test_case.assertTrue(__version__ == flow.__version__)

        mock.disable()
        from torch import nn
        from torch.version import __version__

        test_case.assertTrue(nn.__package__ == "torch.nn")
        test_case.assertTrue(__version__ == torch.__version__)

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
        _import_both()
        with mock.enable():
            test_case.assertTrue(torch.__package__ == "oneflow")
            with mock.disable():
                test_case.assertTrue(torch.__package__ == "torch")
            test_case.assertTrue(torch.__package__ == "oneflow")
        with mock.disable():
            test_case.assertTrue(torch.__package__ == "torch")
            with mock.enable():
                test_case.assertTrue(torch.__package__ == "oneflow")
            test_case.assertTrue(torch.__package__ == "torch")


if __name__ == "__main__":
    unittest.main()
