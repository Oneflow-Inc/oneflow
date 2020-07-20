# __init__.py, rename to avoid being added to PYTHONPATH
from __future__ import absolute_import

from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.core.job.job_conf_pb2 import JobConfigProto
import oneflow.python.framework.session_util as session_util
del session_util

import oneflow.python.framework.dtype as dtype
for x in dir(dtype):
    if x.startswith('_') == False: locals()[x] = getattr(dtype, x)
del x
del dtype

import oneflow.python.framework.register_python_callback

import atexit
import oneflow.python.framework.c_api_util
atexit.register(oneflow.python.framework.c_api_util.DestroyEnv)
atexit.register(oneflow.python.framework.session_context.TryCloseDefaultSession)
del atexit

del absolute_import
del oneflow
del python
#del core
