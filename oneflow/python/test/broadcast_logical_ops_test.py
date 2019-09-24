import oneflow as flow
import numpy as np

@flow.function
def EqualJob(a=flow.input_blob_def((5, 2)), b=flow.input_blob_def((5, 2))):
    flow.config.default_data_type(flow.float)
    return a == b

@flow.function
def NotEqualJob(a=flow.input_blob_def((5, 2)), b=flow.input_blob_def((5, 2))):
    flow.config.default_data_type(flow.float)
    return a != b

@flow.function
def LessJob(a=flow.input_blob_def((5, 2)), b=flow.input_blob_def((5, 2))):
    flow.config.default_data_type(flow.float)
    return a < b

@flow.function
def LessEqualJob(a=flow.input_blob_def((5, 2)), b=flow.input_blob_def((5, 2))):
    flow.config.default_data_type(flow.float)
    return a <= b

@flow.function
def GreaterJob(a=flow.input_blob_def((5, 2)), b=flow.input_blob_def((5, 2))):
    flow.config.default_data_type(flow.float)
    return a > b

@flow.function
def GreaterEqualJob(a=flow.input_blob_def((5, 2)), b=flow.input_blob_def((5, 2))):
    flow.config.default_data_type(flow.float)
    return a >= b

x = np.random.rand(5, 2).astype(np.float32)
y = np.random.rand(5, 2).astype(np.float32)
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

