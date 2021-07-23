import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList


def _test_meshgrid_forawd(test_case, device):
    input1 = flow.Tensor(
        np.array([1, 2, 3]), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.array([4, 5, 6]), dtype=flow.float32, device=flow.device(device)
    )
    (np_x, np_y) = np.meshgrid(input1.numpy(), input2.numpy(), indexing="ij")
    (of_x, of_y) = flow.meshgrid(input1, input2)
    test_case.assertTrue(np.allclose(of_x.numpy(), np_x, 0.0001, 0.0001))
    test_case.assertTrue(np.allclose(of_y.numpy(), np_y, 0.0001, 0.0001))


def _test_meshgrid_forawd_scalr(test_case, device):
    input1 = flow.Tensor(np.array(1.0), dtype=flow.float32, device=flow.device(device))
    input2 = flow.Tensor(np.array(2.0), dtype=flow.float32, device=flow.device(device))
    (np_x, np_y) = np.meshgrid(input1.numpy(), input2.numpy(), indexing="ij")
    (of_x, of_y) = flow.meshgrid(input1, input2)
    test_case.assertTrue(np.allclose(of_x.numpy(), np_x, 0.0001, 0.0001))
    test_case.assertTrue(np.allclose(of_y.numpy(), np_y, 0.0001, 0.0001))


def _test_meshgrid_forawd_3tensor(test_case, device):
    input1 = flow.Tensor(
        np.array([1, 2, 3]), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.array([4, 5, 6]), dtype=flow.float32, device=flow.device(device)
    )
    input3 = flow.Tensor(
        np.array([7, 8, 9]), dtype=flow.float32, device=flow.device(device)
    )
    (np_x, np_y, np_z) = np.meshgrid(
        input1.numpy(), input2.numpy(), input3.numpy(), indexing="ij"
    )
    (of_x, of_y, of_z) = flow.meshgrid(input1, input2, input3)
    test_case.assertTrue(np.allclose(of_x.numpy(), np_x, 0.0001, 0.0001))
    test_case.assertTrue(np.allclose(of_y.numpy(), np_y, 0.0001, 0.0001))
    test_case.assertTrue(np.allclose(of_z.numpy(), np_z, 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestMeshGrid(flow.unittest.TestCase):
    def test_meshgrid(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_meshgrid_forawd,
            _test_meshgrid_forawd_scalr,
            _test_meshgrid_forawd_3tensor,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
