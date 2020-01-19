import oneflow as flow
import numpy as np
from oneflow.python.framework.dtype import convert_of_dtype_to_numpy_dtype

flow.env.ctrl_port(20099)
of_dtype = flow.int8
dtype = convert_of_dtype_to_numpy_dtype(of_dtype)
flow.config.default_data_type(of_dtype)
shape = (512,)


def input_blob_def():
    return flow.input_blob_def(shape, dtype=of_dtype, is_dynamic=True)


@flow.function
def EqualJob(a=input_blob_def()):
    flow.config.default_data_type(of_dtype)
    b = flow.constant_scalar(value=0, dtype=flow.int8)
    equal = a == b
    return (equal, flow.math.reduce_sum(equal))


x = np.random.randint(0, 2, shape).astype(dtype)
y = EqualJob(x).get()
y, cnt = y

a = x == 0
b = y.ndarray()
assert (a == b).all(), "{} vs {}".format(a, b)

a = cnt.ndarray().item()
b = np.sum(y.ndarray())
assert a == b, "{} vs {}".format(a, b)

a = cnt.ndarray().item()
b = np.sum(x == 0)
assert a == b, "{} vs {}".format(a, b)
