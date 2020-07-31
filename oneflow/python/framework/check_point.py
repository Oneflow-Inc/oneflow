"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import datetime
import os

import numpy as np
import oneflow.python.framework.hob as hob
import oneflow.python.framework.job_instance as job_instance
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export
from typing import List, Union


@oneflow_export("train.CheckPoint")
class CheckPoint(object):
    """Create a `CheckPoint` object to manage checkpoint manually.

    """

    def __init__(self) -> None:
        pass

    @session_ctx.try_init_default_session
    def save(self, path: str) -> None:
        r"""save a checkpoint to `path`.

        Args:
            path: A `string` of path to save checkpoint. 
        """
        assert type(path) is str
        enable_if.unique([lazy_checkpoint_save, eager_checkpoint_save])(path)

    @session_ctx.try_init_default_session
    def init(self) -> None:
        r"""Initialize models by default initializer of op or Job.
        """
        enable_if.unique([lazy_checkpoint_init, eager_checkpoint_init])()

    @session_ctx.try_init_default_session
    def load(self, path: str) -> None:
        r"""load a checkpoint from `path` and initialize models.

        Args:
            path: A `string` of path to load checkpoint.
        """
        assert type(path) is str
        enable_if.unique([lazy_checkpoint_load, eager_checkpoint_load])(path)


@enable_if.condition(hob.in_normal_mode & ~hob.eager_execution_enabled)
def lazy_checkpoint_save(path):
    session_ctx.GetDefaultSession().LaunchJob(_MakeModelSaveJobFunc(path))


@enable_if.condition(hob.in_normal_mode & ~hob.eager_execution_enabled)
def lazy_checkpoint_init():
    session_ctx.GetDefaultSession().LaunchJob(_MakeModelInitJobFunc())


@enable_if.condition(hob.in_normal_mode & ~hob.eager_execution_enabled)
def lazy_checkpoint_load(path):
    session_ctx.GetDefaultSession().LaunchJob(_MakeModelLoadJobFunc(path))


@enable_if.condition(hob.in_normal_mode & hob.eager_execution_enabled)
def eager_checkpoint_save(path):
    raise NotImplementedError


@enable_if.condition(hob.in_normal_mode & hob.eager_execution_enabled)
def eager_checkpoint_init():
    # eager variables are initialized in oneflow.get_variable()
    pass


@enable_if.condition(hob.in_normal_mode & hob.eager_execution_enabled)
def eager_checkpoint_load(path):
    session_ctx.GetDefaultSession().snapshot_mgr.load(path)


def _MakeModelInitJobFunc():
    def push_cb(blob):
        pass

    def finish_cb():
        pass

    sess = session_ctx.GetDefaultSession()
    return job_instance.MakeJobInstance(
        str(sess.inter_user_job_info.global_model_init_job_name),
        push_cb=push_cb,
        finish_cb=finish_cb,
    )


def _MakeModelLoadJobFunc(path):
    def push_cb(blob):
        blob.CopyFromNdarray(np.frombuffer(path.encode("ascii"), dtype=np.int8))

    def finish_cb():
        pass

    sess = session_ctx.GetDefaultSession()
    return job_instance.MakeJobInstance(
        str(sess.inter_user_job_info.global_model_load_job_name),
        push_cb=push_cb,
        finish_cb=finish_cb,
    )


def _MakeModelSaveJobFunc(path):
    def push_cb(blob):
        blob.CopyFromNdarray(np.frombuffer(path.encode("ascii"), dtype=np.int8))

    def finish_cb():
        pass

    sess = session_ctx.GetDefaultSession()
    return job_instance.MakeJobInstance(
        str(sess.inter_user_job_info.global_model_save_job_name),
        push_cb=push_cb,
        finish_cb=finish_cb,
    )


@oneflow_export("train.SimpleCheckPointManager")
class SimpleCheckPointManager(object):
    r"""`SimpleCheckPointManager` is a simple automatic checkpoint manager.

    Args:
        root_path: root path of snapshot
        prefix: prefix of snapshot
    """

    def __init__(self, root_path: str, prefix: str = "snapshot_") -> None:
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        else:
            assert os.path.isdir(root_path)
        self._root_path = root_path
        self._prefix = prefix
        self._checkpoint = CheckPoint()

    def list_checkpoints(self) -> List[str]:
        def is_snapshot(name):
            if not name.startswith(self._prefix):
                return False
            snapshot_done = os.path.join(self._GetSnapshotPath(name), "snapshot_done")
            return os.path.exists(snapshot_done) and os.path.isfile(snapshot_done)

        return sorted([f for f in os.listdir(self._root_path) if is_snapshot(f)])

    def latest_checkpoint(self) -> Union[str, None]:
        names = self.list_checkpoints()
        if not names:
            return None
        else:
            return names[-1]

    def initialize_or_restore(self) -> None:
        name = self.latest_checkpoint()
        if name:
            self._checkpoint.load(self._GetSnapshotPath(name))
        else:
            self._checkpoint.init()
            self.save()

    def save(self) -> None:
        self._checkpoint.save(self._GetSnapshotPath(self._NextSnapshotName()))

    def _NextSnapshotName(self) -> str:
        return self._prefix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _GetSnapshotPath(self, name: str) -> str:
        return os.path.join(self._root_path, name)


class SnapshotManager(object):
    def __init__(self):
        self.name2path_ = dict()

    def load(self, root_dir, refresh=True):
        assert os.path.isdir(root_dir)

        if refresh:
            self.name2path_ = dict()

        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)
            if not os.path.isdir(file_path):
                continue

            has_out_subfile = False
            for f in os.listdir(file_path):
                fpath = os.path.join(file_path, f)
                if f == "out" and os.path.isfile(fpath):
                    has_out_subfile = True

            if not has_out_subfile:
                continue

            assert file not in self.name2path_
            self.name2path_[file] = os.path.join(file_path, "out")

    def get_snapshot_path(self, name):
        try:
            return self.name2path_[name]
        except KeyError:
            return None
