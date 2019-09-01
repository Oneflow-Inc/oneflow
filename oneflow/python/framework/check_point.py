from __future__ import absolute_import

import oneflow.python.framework.runtime_context as runtime_ctx
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('train.CheckPoint')
class CheckPoint(object):
    def __init__(self):
        pass

    def restore(self, save_path=None):
        return CheckPointRestoreStatus(save_path)

    def save(self, file_prefix=None, session = None):
        assert file_prefix == None
        if session == None: session = runtime_ctx.default_session
        session.NoReturnRun(_MakeModelSaveJobFunc())

class CheckPointRestoreStatus(object):
    def __init__(self, save_path):
        pass

    def initialize_or_restore(self, session = None):
        if session == None: session = runtime_ctx.default_session
        session.NoReturnRun(_MakeModelInitJobFunc())

def _MakeModelInitJobFunc():
    def ModelInit():
        pass
    ModelInit.__name__ = str(runtime_ctx.inter_user_job_info.global_model_init_job_name)
    ModelInit.__oneflow_input_blob_defs__ = ()
    ModelInit.__oneflow_output_remote_blobs__ = None
    return ModelInit
    
def _MakeModelSaveJobFunc():
    def ModelSave():
        pass
    ModelSave.__name__ = str(runtime_ctx.inter_user_job_info.global_model_save_job_name)
    ModelSave.__oneflow_input_blob_defs__ = ()
    ModelSave.__oneflow_output_remote_blobs__ = None
    return ModelSave
