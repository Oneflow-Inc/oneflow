import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.attr_util import (
    convert_to_user_attr_value,
)


def user_op_expr_call(self, *args, **kwargs):
    args = list(args)
    for i in range(len(args)):
        arg = args[i]
        if isinstance(arg, flow.Tensor):
            if not arg.is_determined:
                arg.determine()
            args[i] = arg._local_or_consistent_tensor
    attrs = oneflow._oneflow_internal.MutableCfgAttrMap()
    for (attr_name, attr_value) in kwargs.items():
        assert isinstance(attr_name, str)
        attrs[attr_name] = convert_to_user_attr_value(
            self.op_type_name, attr_name, attr_value
        )
    try:
        results = self.apply(args, attrs)
    except oneflow._oneflow_internal.exception.Exception:
        raise oneflow._oneflow_internal.exception.GetThreadLocalLastError()
    return results


def RegisterMethod4UserOpExpr():
    oneflow._oneflow_internal.one.UserOpExpr.__call__ = user_op_expr_call
