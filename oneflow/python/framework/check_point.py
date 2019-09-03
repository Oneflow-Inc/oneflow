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

    def save(self, path, session=None):
        assert type(path) is str
        session.NoReturnRun(_MakeModelSaveJobFunc(path))

    def init(self, session=None):
        c_api_util.LaunchJob(_MakeModelInitJobFunc())

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
    def push_cb(blob):
        pass

    def finish_cb():
        pass

    return job_instance.MakeJobInstance(str(runtime_ctx.inter_user_job_info.global_model_init_job_name),
                                        push_cb=push_cb,
                                        finish_cb=finish_cb)


def _MakeModelLoadJobFunc(path):
    def push_cb(blob):
        blob.CopyFromNdarray(np.asarray(list(map(int, path.encode('ascii'))), dtype=np.int8))

    def finish_cb():
        pass

    return job_instance.MakeJobInstance(str(runtime_ctx.inter_user_job_info.global_model_load_job_name),
                                        push_cb=push_cb,
                                        finish_cb=finish_cb)


def _MakeModelSaveJobFunc(path):
    def push_cb(blob):
        blob.CopyFromNdarray(np.asarray(list(map(int, path.encode('ascii'))), dtype=np.int8))

    def finish_cb():
        pass

    return job_instance.MakeJobInstance(str(runtime_ctx.inter_user_job_info.global_model_save_job_name),
                                        push_cb=push_cb,
                                        finish_cb=finish_cb)
