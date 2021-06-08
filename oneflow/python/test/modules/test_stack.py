import oneflow as flow

flow.experimental.enable_eager_execution()
from test_util import GenArgList
import numpy as np
import unittest
from collections import OrderedDict


def _test_stack(test_case, device):
    x = np.ones(shape=(2, 4, 6), dtype=np.float32)
    y = np.ones(shape=(2, 4, 6), dtype=np.float32)
    x_tensor = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    y_tensor = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    np_out = np.stack([x, y], axis=1)
    flow_out = flow.experimental.stack([x_tensor, y_tensor], dim=1).numpy()
    assert np.array_equal(np_out, flow_out)


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)


class TestStack(flow.unittest.TestCase):
    def test_stack(test_case):
        arg_dict = OrderedDict()
        # arg_dict["test_fun"] = [_test_masked_fill, _test_masked_fill_backward]
        arg_dict["test_fun"] = [_test_stack, ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == '__main__':
    unittest.main()
