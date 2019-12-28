import oneflow as flow
import numpy as np

def test_FixedTensorDef(test_case):
    @flow.function()
    def Foo(x=flow.FixedTensorDef((2, 5))): return x
    data = np.ones((2, 5), dtype=np.float32)
    of_ret =Foo(data).get()
    test_case.assertEqual(of_ret.ndarray().max(), 1)
    test_case.assertEqual(of_ret.ndarray().min(), 1)
    test_case.assertTrue(np.allclose(of_ret.ndarray(), data))
    
def test_FixedTensorDef_2_device(test_case):
    flow.config.gpu_device_num(2)
    @flow.function()
    def Foo(x=flow.FixedTensorDef((2, 5))): return x
    data = np.ones((2, 5), dtype=np.float32)
    of_ret =Foo(data).get()
    test_case.assertEqual(of_ret.ndarray().max(), 1)
    test_case.assertEqual(of_ret.ndarray().min(), 1)
    test_case.assertTrue(np.allclose(of_ret.ndarray(), data))

def test_MirroredTensorDef(test_case):
    @flow.function()
    def Foo(x=flow.MirroredTensorDef((2, 5))): return x
    data = np.ones((1, 5), dtype=np.float32)
    ndarray_list = Foo([data]).get().ndarray_list()
    test_case.assertEqual(len(ndarray_list), 1)
    test_case.assertTrue(np.allclose(ndarray_list[0], data))

def test_MirroredTensorListDef(test_case):
    @flow.function()
    def Foo(x=flow.MirroredTensorListDef((2, 5))): return x
    data = np.ones((1, 5), dtype=np.float32)
    ndarray_list = Foo([[data]]).get().ndarray_lists()
    test_case.assertEqual(len(ndarray_list), 1)
    test_case.assertEqual(len(ndarray_list[0]), 1)
    test_case.assertTrue(np.allclose(ndarray_list[0][0], data))
