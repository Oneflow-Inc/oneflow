from __future__ import absolute_import

from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.core.job.job_pb2 import JobConfigProto

import oneflow.python.framework.session_util as session_util
del session_util

import oneflow.python.framework.dtype as dtype

import inspect

for x in dir(dtype):
    if x.startswith('_') == False: locals()[x] = getattr(dtype, x)
del x
del dtype

import oneflow.python.__export_symbols__
import oneflow.python.oneflow_export as oneflow_export
for field, api in oneflow_export.exported._SubApi().items(): locals()[field] = api
for field, func_or_class in vars(oneflow_export.exported).items():
    if inspect.isfunction(func_or_class) or inspect.isclass(func_or_class):
        locals()[field] = func_or_class
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
