from __future__ import absolute_import

import oneflow.python.framework.runtime_context as runtime_ctx
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.job_instance as job_instance
import numpy as np


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

    def load(self, path, session=None):
        assert type(path) is str
        c_api_util.LaunchJob(_MakeModelLoadJobFunc(path))


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
    ModelInit.__oneflow_input_blob_defs__ = ()
    ModelInit.__oneflow_output_remote_blobs__ = None
    return ModelInit


def _MakeModelLoadJobFunc(path):
    def push_cb(blob):
        byte_list = list(map(int, path.encode('ascii')))
        arr = np.zeros(65536, np.uint8)
        arr[0:len(byte_list)] = np.asarray(byte_list, dtype=np.uint8)
        blob.CopyFromNdarray(arr)

    def finish_cb():
        pass

    return job_instance.MakeJobInstance(str(runtime_ctx.inter_user_job_info.global_model_load_job_name),
                                        push_cb=push_cb,
                                        finish_cb=finish_cb)


def _MakeModelSaveJobFunc():
    def ModelSave():
        pass

    ModelSave.__name__ = str(runtime_ctx.inter_user_job_info.global_model_save_job_name)
    ModelSave.__oneflow_input_blob_defs__ = ()
    ModelSave.__oneflow_output_remote_blobs__ = None
    return ModelSave
