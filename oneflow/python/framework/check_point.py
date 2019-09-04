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

    def save(self, path):
        assert type(path) is str
        c_api_util.LaunchJob(_MakeModelSaveJobFunc(path))

    def init(self):
        c_api_util.LaunchJob(_MakeModelInitJobFunc())

    def load(self, path):
        assert type(path) is str
        c_api_util.LaunchJob(_MakeModelLoadJobFunc(path))


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
