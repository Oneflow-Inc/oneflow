from __future__ import absolute_import

import functools
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("function")
def function(job_func):
    r"""Creates a callable OneFlow graph from a Python function.
    For instance::
    
        @oneflow.function
        def train(
            image_blob=oneflow.input_blob_def(
                shape=(2, 255, 255, 3), dtype=flow.float32, is_dynamic=True
            )
        ):
            # your network

    Args:
        job_func: job function to be compiled. Usually the function is decorated by decorator `@oneflow.function`
    Returns:
        If func is not None, returns a callable that will execute the compiled 
        function (and return zero or more Blob objects). 
        If func is None, returns a decorator that, when invoked with a single 
        func argument, returns a callable equivalent to the case above.
    """
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
