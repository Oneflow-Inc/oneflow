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
    return flow._oneflow_internal.GetCurrentStack(10)


@flow.unittest.skip_unless_1n1d()
def test_error_reported_in_thread():
    """
    Stack:
    
      File "oneflow/test/misc/test_stack_getter.py", line 20, in f
        return flow._oneflow_internal.GetCurrentStack(10)
      File "oneflow/test/misc/test_stack_getter.py", line 14, in g
        res = f()
      File "oneflow/test/misc/test_stack_getter.py", line 6, in h
        return g()
      File "oneflow/test/misc/test_stack_getter.py", line 25, in test_error_reported_in_thread
        stack = h()
      File "ckages/_pytest/python.py", line 192, in pytest_pyfunc_call
        result = testfunction(**testargs)
    """
    stack = h()
    assert "flow._oneflow_internal.GetCurrentStack" in stack
    assert "g()" in stack
    assert "f()" in stack
    assert "line 14, in g" in stack


if __name__ == "__main__":
    test_error_reported_in_thread()
