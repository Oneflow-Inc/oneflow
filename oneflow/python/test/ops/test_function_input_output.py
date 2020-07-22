import numpy as np
import oneflow as flow
import oneflow.typing as oft
from typing import Tuple


def test_FixedTensorDef(test_case):
    @flow.global_function()
    def Foo(x: oft.Numpy.Placeholder((2, 5))):
        return x

    data = np.ones((2, 5), dtype=np.float32)
    of_ret = Foo(data).get()
    test_case.assertEqual(of_ret.numpy().max(), 1)
    test_case.assertEqual(of_ret.numpy().min(), 1)
    test_case.assertTrue(np.allclose(of_ret.numpy(), data))


def test_FixedTensorDef_batch_axis(test_case):
    @flow.global_function()
    def Foo(x: oft.Numpy.Placeholder((2, 5), batch_axis=1)):
        test_case.assertEqual(x.batch_axis, 1)
        return x

    data = np.ones((2, 5), dtype=np.float32)
    Foo(np.ones((2, 5), dtype=np.float32)).get()


def test_FixedTensorDef_no_batch_axis(test_case):
    @flow.global_function()
    def Foo(x: oft.Numpy.Placeholder((2, 5), batch_axis=None)):
        test_case.assertTrue(x.batch_axis is None)
        return x

    data = np.ones((2, 5), dtype=np.float32)
    Foo(np.ones((2, 5), dtype=np.float32)).get()


def test_FixedTensorDef_2_device(test_case):
    flow.config.gpu_device_num(2)

    @flow.global_function()
    def Foo(x: oft.Numpy.Placeholder((2, 5))):
        return x

    data = np.ones((2, 5), dtype=np.float32)
    of_ret = Foo(data).get()
    test_case.assertEqual(of_ret.numpy().max(), 1)
    test_case.assertEqual(of_ret.numpy().min(), 1)
    test_case.assertTrue(np.allclose(of_ret.numpy(), data))


def test_MirroredTensorDef(test_case):
    @flow.global_function()
    def Foo(x: oft.ListNumpy.Placeholder((2, 5))):
        return x

    data = np.ones((1, 5), dtype=np.float32)
    ndarray_list = Foo([data]).get().numpy_list()
    test_case.assertEqual(len(ndarray_list), 1)
    test_case.assertTrue(np.allclose(ndarray_list[0], data))


def test_MirroredTensorListDef(test_case):
    @flow.global_function()
    def Foo(x: oft.ListListNumpy.Placeholder((2, 5))):
        return x

    data = np.ones((1, 5), dtype=np.float32)
    ndarray_list = Foo([[data]]).get().numpy_lists()
    test_case.assertEqual(len(ndarray_list), 1)
    test_case.assertEqual(len(ndarray_list[0]), 1)
    test_case.assertTrue(np.allclose(ndarray_list[0][0], data))


def test_MirroredTensorDef_4_device(test_case):
    num_gpus = 4
    flow.config.gpu_device_num(num_gpus)

    image_shape = (64, 3, 224, 224)
    label_shape = (64, 1)

    @flow.global_function()
    def Foo(
        image_label: Tuple[
            oft.ListNumpy.Placeholder(image_shape),
            oft.ListNumpy.Placeholder(label_shape),
        ]
    ):
        return image_label

    ndarray_lst = lambda shape: [
        np.random.rand(*shape).astype(np.float32) for i in range(num_gpus)
    ]
    images = ndarray_lst(image_shape)
    labels = ndarray_lst(label_shape)
    inputs = (images, labels)

    outputs = [output.numpy_list() for output in Foo(inputs).get()]
    test_case.assertEqual(len(outputs), len(inputs))
    for o, i in zip(outputs, inputs):
        test_case.assertEqual(len(o), len(i))
        for o_nda, i_nda in zip(o, i):
            assert type(o_nda) is np.ndarray
            assert type(i_nda) is np.ndarray
            # test_case.assertTrue(np.allclose(o_nda, i_nda))
            test_case.assertTrue(np.array_equal(o_nda, i_nda))
