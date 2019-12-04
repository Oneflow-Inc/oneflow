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

import oneflow.python.__export_symbols__
import oneflow.python.oneflow_export as oneflow_export
for object_name in oneflow_export.exported_object_names:
    locals()[object_name] = getattr(oneflow_export.exported_objects, object_name)
if 'object_name' in locals(): del object_name
del oneflow_export

import atexit
import oneflow.python.framework.c_api_util
atexit.register(oneflow.python.framework.c_api_util.DestroyEnv)
atexit.register(oneflow.python.framework.session_context.TryCloseDefaultSession)
del atexit

del absolute_import
del python
#del core
