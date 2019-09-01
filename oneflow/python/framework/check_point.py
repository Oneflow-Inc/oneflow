from __future__ import absolute_import

import oneflow.python.framework.runtime_context as runtime_ctx
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export('train.CheckPoint')
class CheckPoint(object):
    def __init__(self):
        pass

    def restore(self, save_path=None):
        return CheckPointRestoreStatus(save_path)

    def save(self, file_prefix=None, session=None):
        assert file_prefix is None
        if session is None:
            session = runtime_ctx.default_session
        session.NoReturnRun(_MakeModelSaveJobFunc())

    def init(self, session=None):
        if session is None:
            session = runtime_ctx.default_session
        session.NoReturnRun(_MakeModelInitJobFunc())

    def load(self, file_prefix=None, session=None):
        assert file_prefix is None
        if session is None:
            session = runtime_ctx.default_session
        session.NoReturnRun(_MakeModelLoadJobFunc())


class CheckPointRestoreStatus(object):
    def __init__(self, save_path):
        pass

    def initialize_or_restore(self, session=None):
        if session is None:
            session = runtime_ctx.default_session
        session.NoReturnRun(_MakeModelInitJobFunc())


def _MakeModelInitJobFunc():
    def ModelInit():
        pass

    ModelInit.__name__ = str(runtime_ctx.inter_user_job_info.global_model_init_job_name)
    ModelInit.__oneflow_input_remote_blobs__ = ()
    ModelInit.__oneflow_output_remote_blobs__ = None
    return ModelInit


def _MakeModelLoadJobFunc():
    def ModelLoad():
        pass

    ModelLoad.__name__ = str(runtime_ctx.inter_user_job_info.global_model_load_job_name)
    ModelLoad.__oneflow_input_remote_blobs__ = ()
    ModelLoad.__oneflow_output_remote_blobs__ = None
    return ModelLoad


def _MakeModelSaveJobFunc():
    def ModelSave():
        pass

    ModelSave.__name__ = str(runtime_ctx.inter_user_job_info.global_model_save_job_name)
    ModelSave.__oneflow_input_remote_blobs__ = ()
    ModelSave.__oneflow_output_remote_blobs__ = None
    return ModelSave
