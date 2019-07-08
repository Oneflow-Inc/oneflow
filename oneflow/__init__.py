from __future__ import absolute_import

from oneflow.python.framework.oneflow import run
from oneflow.python.framework.decorator_util import remote
from oneflow.python.framework.decorator_util import static_assert
from oneflow.python.framework.decorator_util import main
from oneflow.python.framework.config import config_resource
from oneflow.python.framework.config import config_io
from oneflow.python.framework.config import config_cpp_flags
from oneflow.python.framework.config import config_profiler
from oneflow.python.framework.compiler import val
from oneflow.python.framework.compiler import var
from oneflow.python.framework.compiler import parse_job_as_func_body
from oneflow.python.framework.inter_user_job import pull

import oneflow.python.framework.dtype as dtype
for x in dir(dtype):
    if x.startswith('_') == False: locals()[x] = getattr(dtype, x)
del x

del absolute_import
del python
del core
del dtype
