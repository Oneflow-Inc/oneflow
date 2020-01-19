/home/caishenghang/oneflow/oneflow/python/test/broadcast_logical_ops_test.pyimport oneflow as flow
import numpy as np
from oneflow.python.framework.dtype import convert_of_dtype_to_numpy_dtype

flow.env.ctrl_port(20099)
of_dtype = flow.int8
dtype = convert_of_dtype_to_numpy_dtype(of_dtype)
flow.config.default_data_type(of_dtype)
shape = (4, 4)


def input_blob_def():
    return flow.input_blob_def(shape, dtype=of_dtype, is_dynamic=True)


@flow.function
def EqualJob(a=input_blob_def()):
    flow.config.default_data_type(of_dtype)
    b = flow.constant_scalar(value=0, dtype=flow.int8)
    return (a == b, flow.math.reduce_sum(a == b))


x = np.random.randint(0, 2, shape).astype(dtype)
print("x = ", x)
print("x == 0 --------------------------------")
y = EqualJob(x).get()
y, cnt = y
assert cnt.ndarray() == np.sum(y.ndarray())
assert cnt.ndarray() == np.sum(x == 0)
