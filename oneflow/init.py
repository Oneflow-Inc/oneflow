# __init__.py, rename to avoid being added to PYTHONPATH
from __future__ import absolute_import

from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.core.job.job_pb2 import JobConfigProto

import oneflow.python.framework.session_util as session_util
del session_util

import oneflow.python.framework.dtype as dtype
for x in dir(dtype):
    if x.startswith('_') == False: locals()[x] = getattr(dtype, x)
del x
del dtype

import traceback
try:
    from oneflow.generated import *
except Exception as _e:
    pass

import oneflow.python.__export_symbols__
import oneflow.python.oneflow_export as oneflow_export
for field, api in oneflow_export.exported._SubApi().items(): locals()[field] = api
if 'field' in locals(): del field
if 'api' in locals(): del api
del oneflow_export

import atexit
import oneflow.python.framework.c_api_util
atexit.register(oneflow.python.framework.c_api_util.DestroyEnv)
atexit.register(oneflow.python.framework.session_context.TryCloseDefaultSession)
del atexit

del absolute_import
del oneflow
del python
#del core
