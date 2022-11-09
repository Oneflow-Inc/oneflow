import unittest
import torch
import numpy as np
import oneflow as flow
import oneflow.unittest
from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgDict


def _test_global_broadcast_tensors(
    test_case, input_shape, other_shape, input_sbp, other_sbp, device
):
    input_np = np.random.randn(*input_shape).astype(np.float32)
    other_np = np.random.randn(*input_shape).astype(np.float32)

    torch_x = torch.tensor(input_np, requires_grad=True)
    torch_other = torch.tensor(other_np, requires_grad=True)

    torch_y, torch_z = torch.broadcast_tensors(torch_x, torch_other)

    torch_y.sum().backward()
    torch_z.sum().backward()

    placement = flow.placement(device, np.array(range(flow.env.get_world_size())))

    flow_x = flow.tensor(input_np, requires_grad=True)
    flow_other = flow.tensor(other_np, requires_grad=True)

    global_x = flow_x.to_global(placement=placement, sbp=flow.sbp.broadcast)
    global_other = flow_other.to_global(placement=placement, sbp=flow.sbp.broadcast)

    if global_x.sbp != input_sbp:
        global_x = global_x.to_global(sbp=input_sbp, grad_sbp=flow.sbp.broadcast)
    if global_other.sbp != other_sbp:
        global_other = global_other.to_global(
            sbp=other_sbp, grad_sbp=flow.sbp.broadcast
        )

    flow_y, flow_z = flow.broadcast_tensors(global_x, global_other)

    flow_y.sum().backward()
    flow_z.sum().backward()

    global_y = flow_y.to_global(sbp=flow.sbp.broadcast)
    global_z = flow_z.to_global(sbp=flow.sbp.broadcast)

    if flow.env.get_rank() == 0:
        torch_y_np = torch_y.detach().cpu().numpy()
        torch_z_np = torch_z.detach().cpu().numpy()
        flow_y_np = global_y.to_local().numpy()
        flow_z_np = global_z.to_local().numpy()

        torch_grad_x = torch_x.grad.cpu().numpy()
        torch_grad_other = torch_other.grad.cpu().numpy()
        flow_grad_x = flow_x.grad.numpy()
        flow_grad_other = flow_other.grad.numpy()

        test_case.assertTrue(np.array_equal(torch_y_np, flow_y_np))
        test_case.assertTrue(np.array_equal(torch_z_np, flow_z_np))
        test_case.assertTrue(np.array_equal(torch_grad_x, flow_grad_x))
        test_case.assertTrue(np.array_equal(torch_grad_other, flow_grad_other))


class TestGlobalBroadcastOps(flow.unittest.TestCase):
    # flow.broadcast_shapes‘s input are shapes, so it can't be tested in global mode
    # flow.broadcast_to is an alias of flow.expand, so its global tests are same as flow.expand's

    def test_global_tensors(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["data"] = [
            ((2, 2), (2, 2, 2), flow.sbp.split(0), flow.sbp.split(0)),
            ((2, 2), (2, 2, 2), flow.sbp.split(0), flow.sbp.split(1)),
            ((2, 2), (2, 2, 2), flow.sbp.split(1), flow.sbp.split(0)),
            ((2, 2), (2, 2, 2), flow.sbp.split(1), flow.sbp.split(1)),
            ((2, 2), (2, 2, 2), flow.sbp.split(0), flow.sbp.broadcast),
            ((2, 2), (2, 2, 2), flow.sbp.broadcast, flow.sbp.split(0)),
            ((2, 2), (2, 2, 2), flow.sbp.broadcast, flow.sbp.broadcast),
        ]
        for kwargs in GenArgDict(arg_dict):
            input_shape, other_shape, input_sbp, other_sbp = kwargs.pop("data")
            _test_global_broadcast_tensors(
                test_case, input_shape, other_shape, input_sbp, other_sbp, **kwargs
            )


if __name__ == "__main__":
    unittest.main()
