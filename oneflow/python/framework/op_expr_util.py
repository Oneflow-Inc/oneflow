import oneflow as flow
import oneflow_api


def user_op_expr_call(self, *args):
    args = list(args)
    for i in range(len(args)):
        arg = args[i]
        if isinstance(arg, flow.Tensor):
            if not arg.is_determined:
                arg.determine()
            args[i] = arg._local_or_consistent_tensor

    results = self.apply(args)
    for i, out in enumerate(results):
        tensor = flow.Tensor(*out.shape)
        tensor._local_or_consistent_tensor = out
        tensor._undetermined_tensor = None
        results[i] = tensor

    return results


def RegisterMethod4UserOpExpr():
    oneflow_api.one.UserOpExpr.__call__ = user_op_expr_call
