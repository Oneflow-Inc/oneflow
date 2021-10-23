import oneflow
import uuid


def trace(f):
    def wrapper(*args, **kwargs):
        m = args[0]
        assert isinstance(m, oneflow.nn.Module)
        for arg in args[1::]:
            print(id(arg))
            isinstance(arg, oneflow._oneflow_internal.Tensor)
        func_name = str(uuid.uuid4()).replace("-", "")
        func_name = f"jit{func_name}"
        print("JIT enabled")
        assert oneflow._oneflow_internal.ir.toggle_jit(func_name)
        oneflow._oneflow_internal.ir.set_jit_forward_args(
            args[1::], list(m.parameters())
        )
        # NOTE: forbid calling __repr__ in the forward function
        result = f(*args, **kwargs)
        # TODO: SetJitForwardResults(result)
        print("JIT disabled")
        assert not oneflow._oneflow_internal.ir.toggle_jit(func_name)
        return result

    return wrapper
