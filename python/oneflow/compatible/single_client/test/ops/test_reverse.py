import unittest
import numpy as np
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp

def _of_reverse(input, axis, dtype):
    flow.clear_default_session()

    @flow.global_function()
    def reverse(input: tp.Numpy.Placeholder(shape=input.shape, dtype=dtype)) -> tp.Numpy:
        return flow.reverse(input, axis)
    return reverse(input)

def _test_reverse(test_case, input, axis, dtype, verbose=False):
    assert isinstance(input, np.ndarray)
    input = input.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
    slice_list = [slice(None)] * input.ndim
    for a in axis:
        if a < 0:
            a += input.ndim
        assert a >= 0 and a < input.ndim
        slice_list[a] = slice(None, None, -1)
    output = input[tuple(slice_list)]
    of_output = _of_reverse(input, axis, dtype)
    if verbose:
        print('input: {}\n{}\n'.format(input.shape, input))
        print('comparing output:\n{}\nvs.\n{}'.format(output, of_output))
    test_case.assertTrue(np.array_equal(output, of_output))

@flow.unittest.skip_unless_1n1d()
class TestReverse(flow.unittest.TestCase):

    def test_reverse_case_1(test_case):
        input = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        _test_reverse(test_case, input, [3], flow.int32)

    def test_reverse_case_2(test_case):
        input = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        _test_reverse(test_case, input, [-1], flow.int32)

    def test_reverse_case_3(test_case):
        input = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        _test_reverse(test_case, input, [1], flow.int32)

    def test_reverse_case_4(test_case):
        input = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        _test_reverse(test_case, input, [-3], flow.int32)

    def test_reverse_case_5(test_case):
        input = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        _test_reverse(test_case, input, [2], flow.float32)

    def test_reverse_case_6(test_case):
        input = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        _test_reverse(test_case, input, [-2], flow.float32)
if __name__ == '__main__':
    unittest.main()