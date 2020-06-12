import inspect
import re

import oneflow.python.lib.core.enable_if as enable_if_util
import oneflow.python.lib.core.traceinfo as traceinfo
from oneflow.python.lib.core.high_order_bool import always_true


def oneflow_export(*api_names, **kwargs):
    if len(kwargs) == 1:
        assert "enable_if" in kwargs
        hob_expr = kwargs["enable_if"]
    else:
        assert len(kwargs) == 0
        hob_expr = always_true
    return _GetOneflowExportDecorator(
        api_names, hob_expr, traceinfo.GetFrameLocationStr(-2)
    )


def _GetOneflowExportDecorator(api_names, hob_expr, location):
    def Decorator(func_or_class):
        func_or_class._ONEFLOW_API = api_names
        return func_or_class

    return Decorator


class OneflowApi(object):
    def __init__(self, api_name=""):
        self.api_name_ = api_name
        self.conditional_functions_ = []
        self.default_func_ = _MakeDefaultFunction(api_name, self.conditional_functions_)
        self.sub_api_ = {}

    def __call__(self, *args, **kwargs):
        matched_func = enable_if_util.GetMatchedFunction(self.conditional_functions_)
        if matched_func is None:
            matched_func = self.default_func_
        return matched_func(*args, **kwargs)

    @property
    def _class(self):
        assert len(self.conditional_functions_) == 1
        func_or_class = self.conditional_functions_[0]
        assert inspect.isclass(func_or_class)
        return func_or_class

    def _SubApi(self):
        return self.sub_api_

    def _AddApiFuncOrClass(self, hob_expr, func_or_class, location):
        func_or_class = _GetInvokerIfIsSpecializedFunctor(func_or_class)
        self.conditional_functions_.append((hob_expr, func_or_class, location))

    def _FindOrCreateSubApi(self, field):
        assert re.match("^[\w]+[_\w\d]*$", field)
        if field in self.sub_api_:
            return self.sub_api_[field]
        sub_api = OneflowApi(self.api_name_ + "." + field)
        self.sub_api_[field] = sub_api
        setattr(self, field, sub_api)
        return sub_api


def _MakeDefaultFunction(api_name, conditional_functions):
    def default(get_failed_info, *args, **kwargs):
        raise NotImplementedError(get_failed_info(api_name))

    return enable_if_util.MakeDefaultFunction(default, conditional_functions)


def _GetInvokerIfIsSpecializedFunctor(func_or_class):
    if (
        inspect.isclass(func_or_class)
        and hasattr(func_or_class, "invoke")
        and hasattr(func_or_class.invoke, "__is_specialization_supported__")
    ):
        return func_or_class.invoke
    return func_or_class


exported = OneflowApi("oneflow")
