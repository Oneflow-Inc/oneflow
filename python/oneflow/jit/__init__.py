import oneflow
import uuid


def trace(f):
    def wrapper(*args, **kwargs):
        assert isinstance(args[0], oneflow.nn.Module)
        func_name = str(uuid.uuid4()).replace("-", "")
        func_name = f"jit{func_name}"
        assert oneflow._oneflow_internal.ir.toggle_jit(func_name)
        print("JIT enabled")
        # TODO: SetJitForwardArgs(args)
        result = f(*args, **kwargs)
        # TODO: SetJitForwardResults(result)
        print("JIT disabled")
        assert not oneflow._oneflow_internal.ir.toggle_jit(func_name)
        return result

    return wrapper
