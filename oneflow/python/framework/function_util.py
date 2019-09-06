from __future__ import absolute_import

import functools
import oneflow.python.framework.job_set_util as job_set_util
import oneflow.python.framework.runtime_context as runtime_ctx
import oneflow.python.framework.session_util as session_util
from oneflow.python.oneflow_export import oneflow_export

_job_name2job_func = {}

@oneflow_export("function")
def function(job_func):
    @functools.wraps(job_func)
    def Func(*args):
        return session_util.GetDefaultSession().run(job_func, *args)
    for x in dir(job_func):
        if x.startswith('__oneflow_'):
            setattr(Func, x, getattr(job_func, x))
    job_set_util.add_job(job_func)
    return Func
