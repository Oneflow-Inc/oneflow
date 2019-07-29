from __future__ import absolute_import

import oneflow.python.framework.runtime_context as runtime_ctx
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('train.CheckPoint')
class CheckPoint(object):
    def __init__(self):
        pass

    def restore(save_path=None):
        return CheckPointRestoreStatus(save_path)

    def save(file_prefix=None, session = None):
        assert file_prefix == None
        if session == None: session = runtime_ctx.default_session
        session.run(_MakeModelSaveJobFunc())

class CheckPointRestoreStatus(object):
    def __init__(self, save_path):
        pass

    def initialize_or_restore(session = None):
        TODO()

def _MakeModelSaveJobFunc():
    def ModelSave():
        pass
    ModelSave.__name__ = runtime_ctx.inter_user_job_info.global_model_save_job_name
    return ModelSave
