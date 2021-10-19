import oneflow


def trace(f):
    def wrapper(*args, **kwargs):
        assert oneflow._oneflow_internal.ir.toggle_jit()
        print("JIT enabled")
        result = f(*args, **kwargs)
        print("JIT disabled")
        assert not oneflow._oneflow_internal.ir.toggle_jit()
        return result

    return wrapper
