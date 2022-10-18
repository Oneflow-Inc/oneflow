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
    f()
    return res


def f():
    return flow._oneflow_internal.GetCurrentStack()


@flow.unittest.skip_unless_1n1d()
def test_error():
    """
    Stack:
    
      File "oneflow/test/misc/test_stack_getter.py", line 14, in g
        res = f()
      File "oneflow/test/misc/test_stack_getter.py", line 6, in h
        return g()
      File "oneflow/test/misc/test_stack_getter.py", line 25, in test_error
        stack = h()
    """
    stack = h()
    assert "return g()" in stack
    assert "res = f()" in stack
    assert "line 14, in g" in stack


if __name__ == "__main__":
    test_error()
