from __future__ import absolute_import

from oneflow.python.framework.session import Session
from oneflow.python.framework.val import val
from oneflow.python.framework.config_util import placement

import oneflow.python.framework.dtype as dtype
for x in dir(dtype):
    if x.startswith('_') == False: locals()[x] = getattr(dtype, x)
del x

del absolute_import
del python
del core
del dtype
