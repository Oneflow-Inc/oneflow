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
import shutil

import numpy as np
from google.protobuf import text_format

import oneflow.python.framework.dtype as dtype_util
import oneflow.python.framework.hob as hob
import oneflow.python.framework.job_instance as job_instance
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.eager.op_executor as op_executor
import oneflow.core.framework.variable_meta_info_pb2 as variable_meta_info_pb
from oneflow.python.experimental import interface_op_read_and_write
from oneflow.python.framework.remote_blob import EagerBlobTrait

from oneflow.python.oneflow_export import oneflow_export
from typing import Dict, List, Union, Sequence, Optional


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
    op_executor.EagerSaveVariableBlob(path)


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


META_INFO_FILENAME = "meta"


class FileBackendVariableBlob:
    def __init__(self, name: str, root_dir: str, dtype: Optional[dtype_util.dtype] = None, shape: Optional[Sequence[int]] = None):
        self.name_ = name
        self.root_dir_ = root_dir
        meta_info_path = os.path.join(self.root_dir_, META_INFO_FILENAME)
        if os.path.exists(meta_info_path):
            meta_infos = variable_meta_info_pb.VariableMetaInfos()
            with open(meta_info_path) as f:
                text_format.Parse(f.read(), meta_infos)
            if name in meta_infos.name2meta_info:
                self.has_meta_info_ = True
            else:
                self.has_meta_info_ = False
        else:
            self.has_meta_info_ = False

        if self.has_meta_info_:
            assert dtype is None and shape is None
            self.shape_ = tuple(meta_infos.name2meta_info[name].shape.dim)
            self.dtype_ = dtype_util.convert_proto_dtype_to_oneflow_dtype(
                meta_infos.name2meta_info[name].data_type
            )
        else:
            if shape is not None and dtype is not None:
                self.shape_ = shape
                self.dtype_ = dtype
                self.has_meta_info_ = True
            elif shape is not None or dtype is not None:
                raise RuntimeError("both or neither of shape and dtype should be None")

    def read_slice_as_numpy(self):
        SLICE_LEN = 8192
        with open(self.file_path_, 'rb') as f:
            while True:
                slice = f.read(SLICE_LEN)
                if not slice:
                    break
                yield np.frombuffer(slice, dtype=dtype_util.convert_oneflow_dtype_to_numpy_dtype(self.dtype))

    @property
    def file_path_(self):
        return os.path.join(self.root_dir_, self.name, "out")

    def _NumpyFriendlyToRepr(self):
        return self.numpy()

    def __repr__(self):
        if self.has_meta_info_:
            return '({}, name="{}", shape={}, dtype={})'.format(
                self._NumpyFriendlyToRepr(), self.name, self.shape, self.dtype
            )
        else:
            return '(variable without meta info, name="{}")'.format(self.name)

    @property
    def name(self):
        return self.name_

    @property
    def shape(self):
        return self.shape_

    @property
    def quant_info(self):
        pass

    @property
    def dtype(self):
        return self.dtype_

    def _IsTooLarge(self):
        return False

    def numpy(self) -> np.ndarray:
        if self._IsTooLarge():
            raise RuntimeError('Blob "{}" is too large'.format(self.name))
        if not self.has_meta_info_:
            raise RuntimeError('The variable "{}" does not have meta info'.format(self.name))
        return np.fromfile(self.file_path_, dtype=dtype_util.convert_oneflow_dtype_to_numpy_dtype(self.dtype)).reshape(self.shape)


@oneflow_export("get_all_variables")
def get_all_variables() -> Dict[str, FileBackendVariableBlob]:
    sess = session_ctx.GetDefaultSession()
    interface_ops = sess.interface_ops
    variables = {}
    for op in interface_ops:
        op_attr = sess.OpAttribute4InterfaceOpName(op)
        if op_attr.op_conf.WhichOneof("op_type") != "variable_conf":
            continue
        variables[op] = interface_op_read_and_write.GetEagerInterfaceBlob(op)
    return variables


@oneflow_export("load")
def load(path):
    var_dict = {}
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):
            var_path = os.path.join(path, f, "out")
            if os.path.isfile(var_path):
                var_dict[f] = FileBackendVariableBlob(f, path)
    return var_dict


def read_slice_from_blob(blob):
    #TODO(daquexian): implement real read_slice_from_blob after dynamic network is implemented
    yield blob.numpy()


@oneflow_export("save")
def save(var_dict, path):
    os.makedirs(path, exist_ok=True)
    meta_infos = variable_meta_info_pb.VariableMetaInfos()
    for name, var in var_dict.items():
        meta_info = meta_infos.name2meta_info[name]
        meta_info.shape.dim[:] = var.shape
        meta_info.data_type = var.dtype.oneflow_proto_dtype
        param_path = os.path.join(path, name, "out")
        os.makedirs(os.path.dirname(param_path), exist_ok=True)
        with open(param_path, 'wb') as f:
            for slice in read_slice_from_blob(var):
                f.write(slice.tobytes())
    with open(os.path.join(path, META_INFO_FILENAME), "w") as f:
        f.write(text_format.MessageToString(meta_infos))


def slice_assign(slice_id, slice, var_name):
    #TODO(daquexian): replace it with real slice_assign
    assert slice_id == 0
    var_blob = interface_op_read_and_write.GetEagerInterfaceBlob(var_name)
    slice = np.reshape(slice, var_blob.shape)
    interface_op_read_and_write.FeedValueToInterfaceBlob(
        var_name, slice
    )


def _FeedValueToVariable(var_name, value_blob):
    if isinstance(value_blob, EagerBlobTrait):
        raise NotImplementedError("Loading value from another blob has not been implemented")
    elif isinstance(value_blob, FileBackendVariableBlob):
        if not value_blob.has_meta_info_:
            var_blob = interface_op_read_and_write.GetEagerInterfaceBlob(var_name)
            value_blob = FileBackendVariableBlob(value_blob.name, value_blob.root_dir_, var_blob.dtype, var_blob.shape)
        for slice_id, slice in enumerate(value_blob.read_slice_as_numpy()):
            slice_assign(slice_id, slice, var_name)
    else:
        raise RuntimeError("Unknown value_blob type: " + type(value_blob).__name__)


@oneflow_export("checkpoint.load_variables")
def load_variables(var_dict, ignore_mismatch=False):
    for name, var in var_dict.items():
        if name in get_all_variables():
            _FeedValueToVariable(name, var)
        else:
            if not ignore_mismatch:
                raise RuntimeError('"{}" is not a variable name'.format(name))
