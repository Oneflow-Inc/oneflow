from __future__ import absolute_import

import functools
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("function")
def function(job_func):
    sess = session_ctx.GetDefaultSession()
    @functools.wraps(job_func)
    def Func(*args):
        return _RunJob(sess, job_func, *args)
    for x in dir(job_func):
        if x.startswith('__oneflow_'): setattr(Func, x, getattr(job_func, x))
    sess.AddJob(job_func)
    return Func

def _RunJob(session, job_func, *args):
    return session.TryInit().Run(job_func, *args)
