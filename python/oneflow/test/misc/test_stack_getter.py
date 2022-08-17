import oneflow as flow
import oneflow.unittest


def h():
    return g()


def g():
    res = f()
    res = f()
    res = f()
    res = f()
    res = f()
    res = f()
    return res


def f():
    return flow._oneflow_internal.GetCurrentStack(10)


@flow.unittest.skip_unless_1n1d()
def test_error_reported_in_thread():
    stack = h()
    print(stack)
    assert "flow._oneflow_internal.GetCurrentStack" in stack
    assert "g()" in stack
    assert "f()" in stack

if __name__ == "__main__":
    test_error_reported_in_thread()
