import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList

def _test_bernoulli(test_case, shape):
    input_arr = np.ones(shape)
    x = flow.Tensor(input_arr, device=flow.device('cpu'))
    y = flow.bernoulli(x)
    test_case.assertTrue(np.allclose(y.numpy(), x.numpy()))

def _test_bernoulli_with_generator(test_case, shape):
    generator = flow.Generator()
    generator.manual_seed(0)
    x = flow.Tensor(np.random.rand(*shape), device=flow.device('cpu'))
    y_1 = flow.bernoulli(x, generator=generator)
    y_1.numpy()
    generator.manual_seed(0)
    y_2 = flow.bernoulli(x, generator=generator)
    test_case.assertTrue(np.allclose(y_1.numpy(), y_2.numpy()))

@flow.unittest.skip_unless_1n1d()
class TestBernoulli(flow.unittest.TestCase):

    def test_bernoulli(test_case):
        arg_dict = OrderedDict()
        arg_dict['test_functions'] = [_test_bernoulli]
        arg_dict['shape'] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])
if __name__ == '__main__':
    unittest.main()