import unittest
import oneflow as flow
import oneflow.unittest
from oneflow.mock_torch import Mock


class TestMock(flow.unittest.TestCase):
    def test_complex(test_case):
        mock = Mock()  # default enable mocking
        import torch

        test_case.assertTrue(torch.__package__ == "oneflow")
        import torch.nn

        test_case.assertTrue(torch.nn.__package__ == "oneflow.nn")
        import torch.version

        test_case.assertTrue(torch.version.__version__ == flow.__version__)
        mock.clean_torch(globals())
        mock.disable()
        import torch

        test_case.assertTrue(torch.__package__ == "torch")
        import torch.nn

        test_case.assertTrue(torch.nn.__package__ == "torch.nn")
        import torch.version

        test_case.assertTrue(torch.version.__version__ == torch.__version__)

        mock.clean_torch(globals())
        mock.enable()
        from torch import nn
        from torch.version import __version__

        test_case.assertTrue(nn.__package__ == "oneflow.nn")
        test_case.assertTrue(__version__ == flow.__version__)
        mock.clean_torch(globals())
        mock.disable()
        from torch import nn
        from torch.version import __version__

        test_case.assertTrue(nn.__package__ == "torch.nn")
        test_case.assertTrue(__version__ == torch.__version__)
        mock.clean_torch(globals())
        mock.enable()
        with test_case.assertRaises(Exception) as context:
            from torch import noexist
        test_case.assertTrue(
            "oneflow.noexist is not implemented" in str(context.exception)
        )
        mock.clean_torch(globals())
        mock.disable()
        with test_case.assertRaises(Exception) as context:
            from torch import noexist
        test_case.assertTrue(
            "cannot import name 'noexist' from 'torch'" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
