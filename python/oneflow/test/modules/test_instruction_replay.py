import unittest
from collections import OrderedDict
import oneflow
import numpy as np
import oneflow as flow
from test_util import GenArgList

def _test_instruction_replay_impl(test_case, device, shape):
    x = flow.Tensor(np.random.rand(*shape), device=flow.device(device))
    y = flow.Tensor(np.random.rand(*shape), device=flow.device(device))
    x.determine()
    y.determine()
    oneflow._oneflow_internal.debug.start_recording_instructions()
    z = x + y
    oneflow._oneflow_internal.debug.end_recording_instructions()
    test_case.assertTrue(np.allclose(z.numpy(), x.numpy() + y.numpy(), 0.0001, 0.0001))
    z.zeros_()
    oneflow._oneflow_internal.debug.replay_instructions()
    test_case.assertTrue(np.allclose(z.numpy(), x.numpy() + y.numpy(), 0.0001, 0.0001))
    oneflow._oneflow_internal.debug.clear_recorded_instructions()

@flow.unittest.skip_unless_1n1d()
class TestIntructionReplay(flow.unittest.TestCase):

    def test_instruction_replay(test_case):
        arg_dict = OrderedDict()
        arg_dict['device'] = ['cpu', 'cuda']
        arg_dict['shape'] = [[2, 3], [1, 10]]
        for arg in GenArgList(arg_dict):
            _test_instruction_replay_impl(test_case, *arg)
if __name__ == '__main__':
    unittest.main()