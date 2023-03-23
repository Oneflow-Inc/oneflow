import oneflow as flow
import numpy as np
import os
import unittest

class TestTensorComplex64(unittest.TestCase):
    def setUp(self):
        self.dtype = flow.cfloat
        self.np_dtype = np.complex64
        self.type_str = "ComplexFloatTensor"
        self.a = [1.0 + 1j, 2.0]
        self.np_a = np.array(self.a, dtype=self.np_dtype)
        self.b = [[1.0 + 1j, 2.0], [1.0, 2.0 - 1j], [-1.0, 1j]]
        self.np_b = np.array(self.b, dtype=self.np_dtype)
        self.c = [
            [3.14 + 2j, 3.14 + 2j],
            [3.14 + 2j, 3.14 + 2j],
            [3.14 + 2j, 3.14 + 2j],
        ]
        self.np_c = np.array(self.c, dtype=self.np_dtype)

    def test_fft(self):
        c = flow.from_numpy(self.np_c)
        print(c.dtype)
        print(flow._C.fft(c, dim=0))

if __name__ == "__main__":
    unittest.main()