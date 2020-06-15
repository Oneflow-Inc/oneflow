import oneflow as flow
import numpy as np


def test_eager_logical_blob(test_case):
    flow.enable_eager_execution()

    @flow.function()
    def Foo():
        x = flow.constant(1, shape=(10,), dtype=flow.int32)
        of_ones = x.numpy_mirrored_list()[0]
        np_ones = np.ones((10,), dtype=np.int32)
        test_case.assertTrue(np.array_equal(np_ones, of_ones))

    Foo()
