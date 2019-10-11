from __future__ import absolute_import

import oneflow.python.framework.runtime_context as runtime_ctx
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.job_instance as job_instance
import oneflow.python.framework.session_util as session_util
import numpy as np
import os
import datetime


@oneflow_export('train.CheckPoint')
class CheckPoint(object):
    def __init__(self):
        pass

    @session_util.try_init_default_session
    def save(self, path):
        assert type(path) is str
        c_api_util.LaunchJob(_MakeModelSaveJobFunc(path))

    @session_util.try_init_default_session
    def init(self):
        c_api_util.LaunchJob(_MakeModelInitJobFunc())

    @session_util.try_init_default_session
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
        blob.CopyFromNdarrayOrNestedNdarrayList(np.frombuffer(path.encode('ascii'), dtype=np.int8))

    def finish_cb():
        pass

    return job_instance.MakeJobInstance(str(runtime_ctx.inter_user_job_info.global_model_load_job_name),
                                        push_cb=push_cb,
                                        finish_cb=finish_cb)


def _MakeModelSaveJobFunc(path):
    def push_cb(blob):
        blob.CopyFromNdarrayOrNestedNdarrayList(np.frombuffer(path.encode('ascii'), dtype=np.int8))

    def finish_cb():
        pass

    return job_instance.MakeJobInstance(str(runtime_ctx.inter_user_job_info.global_model_save_job_name),
                                        push_cb=push_cb,
                                        finish_cb=finish_cb)


@oneflow_export('train.SimpleCheckPointManager')
class SimpleCheckPointManager(object):
    def __init__(self, root_path, prefix='snapshot_'):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        else:
            assert os.path.isdir(root_path)
        self._root_path = root_path
        self._prefix = prefix
        self._checkpoint = CheckPoint()

    def list_checkpoints(self):
        def is_snapshot(name):
            if not name.startswith(self._prefix):
                return False
            snapshot_done = os.path.join(self._GetSnapshotPath(name), 'snapshot_done')
            return os.path.exists(snapshot_done) and os.path.isfile(snapshot_done)

        return sorted([f for f in os.listdir(self._root_path) if is_snapshot(f)])

    def latest_checkpoint(self):
        names = self.list_checkpoints()
        if not names:
            return None
        else:
            return names[-1]

    def initialize_or_restore(self):
        name = self.latest_checkpoint()
        if name:
            self._checkpoint.load(self._GetSnapshotPath(name))
        else:
            self._checkpoint.init()
            self.save()

    def save(self):
        self._checkpoint.save(self._GetSnapshotPath(self._NextSnapshotName()))

    def _NextSnapshotName(self):
        return self._prefix + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    def _GetSnapshotPath(self, name):
        return os.path.join(self._root_path, name)
