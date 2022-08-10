import oneflow as flow

def g(x):
    flow._C.throw_error(x)

def f(x):
    x = x.relu()
    g(x)

x = flow.ones(3, 3, 4)
f(x)
