import oneflow
import uuid


def trace(f):
    def wrapper(*args, **kwargs):
        func_name = str(uuid.uuid4()).replace("-", "")
        func_name = f"jit{func_name}"
        assert oneflow._oneflow_internal.ir.toggle_jit(func_name)
        print("JIT enabled")
        result = f(*args, **kwargs)
        print("JIT disabled")
        assert not oneflow._oneflow_internal.ir.toggle_jit(func_name)
        return result

    return wrapper
