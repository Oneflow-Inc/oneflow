from __future__ import absolute_import

import oneflow.python.framework.session_context as session_ctx


def GetDefaultBackwardBlobRegister():
    return session_ctx.GetDefaultSession().backward_blob_register
