from __future__ import absolute_import

import oneflow.python.lib.core.func_inspect_util as func_inspect_util
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('append_func_to_list')
def append_func_to_list(l):
    def Decorator(func):
        def Func(*argv): return func(*argv)
        Func.__name__ = func.__name__
        Func.__oneflow_arg_default__ = func_inspect_util.GetArgDefaults(func)
        l.append(Func)
        return Func
    return Decorator
