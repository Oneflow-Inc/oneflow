from __future__ import absolute_import

import functools
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("function")
def function(job_func):
    @functools.wraps(job_func)
    def Func(*args):
        return session_ctx.GetDefaultSession().Run(job_func, *args)
    for x in dir(job_func):
        if x.startswith('__oneflow_'): setattr(Func, x, getattr(job_func, x))
    session_ctx.GetDefaultSession().AddJob(job_func)
    return Func
