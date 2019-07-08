from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.decorator_context as decorator_context
import oneflow.python.framework.oneflow_mode as oneflow_mode

def remote(func):
    TODO()

def static_assert(func):
    TODO()

def main(func):
    def Main(*arg):
        func(*arg)
    decorator_context.main_func = Main
    if hasattr(func, '__config__func__'):
        Main.__config__func__ = func.__config__func__
    else:
        def _EmptyConfig(job_set):
            pass
        Main.__config__func__ = _EmptyConfig
    return Main;
