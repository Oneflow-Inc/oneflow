import inspect
import re

import oneflow.python.lib.core.enable_if as enable_if_util
import oneflow.python.lib.core.traceinfo as traceinfo
from oneflow.python.lib.core.high_order_bool import always_true


def oneflow_export(*api_names, **kwargs):
    def Decorator(func_or_class):
        func_or_class._ONEFLOW_API = api_names
        return func_or_class

    return Decorator
