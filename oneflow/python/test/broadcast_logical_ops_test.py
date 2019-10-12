import oneflow as flow
import numpy as np
from oneflow.python.framework.dtype import convert_of_dtype_to_numpy_dtype

of_dtype = flow.int32
dtype = convert_of_dtype_to_numpy_dtype(of_dtype)
flow.config.default_data_type(of_dtype)
shape = (5, 2)
def input_blob_def(): return flow.input_blob_def(shape, dtype=of_dtype)

@flow.function
def EqualJob(a=input_blob_def(), b=input_blob_def()):
    flow.config.default_data_type(of_dtype)
    return a == b

@flow.function
def NotEqualJob(a=input_blob_def(), b=input_blob_def()):
    flow.config.default_data_type(of_dtype)
    return a != b

@flow.function
def LessJob(a=input_blob_def(), b=input_blob_def()):
    flow.config.default_data_type(of_dtype)
    return a < b

@flow.function
def LessEqualJob(a=input_blob_def(), b=input_blob_def()):
    flow.config.default_data_type(of_dtype)
    return a <= b

@flow.function
def GreaterJob(a=input_blob_def(), b=input_blob_def()):
    flow.config.default_data_type(of_dtype)
    return a > b

@flow.function
def GreaterEqualJob(a=input_blob_def(), b=input_blob_def()):
    flow.config.default_data_type(of_dtype)
    return a >= b

@flow.function
def LogicalAndJob(a=input_blob_def(), b=input_blob_def()):
    flow.config.default_data_type(of_dtype)
    return flow.math.logical_and(a, b)

x = np.random.randint(0, 2, shape).astype(dtype)
y = np.random.randint(0, 2, shape).astype(dtype)
z = None
print('x = ', x)
print('y = ', y)
print('x == y --------------------------------')
z = EqualJob(x, y).get()
print (z)
print('x == x --------------------------------')
z = EqualJob(x, x).get()
print (z)

print('x != y --------------------------------')
z = NotEqualJob(x, y).get()
print (z)
print('x != x --------------------------------')
z = NotEqualJob(x, x).get()
print (z)

print('x > y --------------------------------')
z = GreaterJob(x, y).get()
print (z)
print('x > x --------------------------------')
z = GreaterJob(x, x).get()
print (z)

print('x >= y --------------------------------')
z = GreaterEqualJob(x, y).get()
print (z)
print('x >= x --------------------------------')
z = GreaterEqualJob(x, x).get()
print (z)

print('x < y --------------------------------')
z = LessJob(x, y).get()
print (z)
print('x < x --------------------------------')
z = LessJob(x, x).get()
print (z)

print('x <= y --------------------------------')
z = LessEqualJob(x, y).get()
print (z)
print('x <= x --------------------------------')
z = LessEqualJob(x, x).get()
print (z)

print('x and y --------------------------------')
z = LogicalAndJob(x, y).get()
print (z)
print('x and x --------------------------------')
z = LogicalAndJob(x, x).get()
print (z)

