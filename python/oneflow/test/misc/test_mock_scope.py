import unittest
import oneflow as flow
import oneflow.unittest
from oneflow.mock_torch import Mock

mock = Mock({})  # default enable mocking


class TestMock(flow.unittest.TestCase):
    def test_complex(test_case):
        import torch

        test_case.assertTrue(torch.__package__ == "oneflow")
        import torch.nn

        test_case.assertTrue(torch.nn.__package__ == "oneflow.nn")
        import torch.version

        test_case.assertTrue(torch.version.__version__ == flow.__version__)
        mock.disable(globals())
        import torch

        test_case.assertTrue(torch.__package__ == "torch")
        import torch.nn

        test_case.assertTrue(torch.nn.__package__ == "torch.nn")
        import torch.version

        test_case.assertTrue(torch.version.__version__ == torch.__version__)

        mock.enable(globals())
        from torch import nn
        from torch.version import __version__

        test_case.assertTrue(nn.__package__ == "oneflow.nn")
        test_case.assertTrue(__version__ == flow.__version__)
        mock.disable(globals())
        from torch import nn
        from torch.version import __version__

        test_case.assertTrue(nn.__package__ == "torch.nn")
        test_case.assertTrue(__version__ == torch.__version__)
        mock.enable(globals())
        with test_case.assertRaises(Exception) as context:
            from torch import noexist
        test_case.assertTrue(
            "oneflow.noexist is not implemented" in str(context.exception)
        )
        mock.disable(globals())
        with test_case.assertRaises(Exception) as context:
            from torch import noexist
        test_case.assertTrue(
            "cannot import name 'noexist' from 'torch'" in str(context.exception)
        )

    def test_with(test_case):
        with mock.enable_with(globals()):
            import torch
            import torch.nn
            import torch.version

            test_case.assertTrue(torch.__package__ == "oneflow")
            test_case.assertTrue(torch.nn.__package__ == "oneflow.nn")
            test_case.assertTrue(torch.version.__version__ == flow.__version__)
        with mock.disable_with(globals()):
            import torch
            import torch.nn
            import torch.version

            test_case.assertTrue(torch.__package__ == "torch")
            test_case.assertTrue(torch.nn.__package__ == "torch.nn")
            test_case.assertTrue(torch.version.__version__ == torch.__version__)


if __name__ == "__main__":
    unittest.main()
