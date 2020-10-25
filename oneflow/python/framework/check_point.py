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

import oneflow
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.ops.initializer_util as initializer_util
import oneflow.python.framework.hob as hob
import oneflow.python.framework.job_instance as job_instance
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.lib.core.async_util as async_util
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.op_executor as op_executor
import oneflow.python.eager.op_infer_util as op_infer_util
import oneflow.core.framework.variable_meta_info_pb2 as variable_meta_info_pb
import oneflow.core.framework.user_op_attr_pb2 as user_op_attr_util
from oneflow.python.experimental import interface_op_read_and_write
from oneflow.python.framework.remote_blob import EagerBlobTrait
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.ops.get_variable as get_variable

from oneflow.python.oneflow_export import oneflow_export
from typing import Dict, List, Union, Sequence, Optional


_lazy_checkpoint = False


@oneflow_export("use_legacy_checkpoint")
def api_use_legacy_checkpoint(val: bool = True) -> None:
    return enable_if.unique([use_legacy_checkpoint])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.any_global_function_defined)
def use_legacy_checkpoint(val=True):
    global _lazy_checkpoint
    _lazy_checkpoint = val


@oneflow_export("legacy_checkpoint_used")
def legacy_checkpoint_used():
    return _lazy_checkpoint


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
        if not legacy_checkpoint_used():
            print(
                "'checkpoint.save()' is deprecated. Please use the new checkpoint API"
            )
        assert type(path) is str
        enable_if.unique([lazy_checkpoint_save, eager_checkpoint_save])(path)

    @session_ctx.try_init_default_session
    def init(self) -> None:
        r"""Initialize models by default initializer of op or Job.
        """
        if not legacy_checkpoint_used():
            print(
                "'checkpoint.init()' is deprecated. It has no effect and will be removed in the future"
            )
        enable_if.unique([lazy_checkpoint_init, eager_checkpoint_init])()

    @session_ctx.try_init_default_session
    def load(self, path: str) -> None:
        r"""load a checkpoint from `path` and initialize models.

        Args:
            path: A `string` of path to load checkpoint.
        """
        if not legacy_checkpoint_used():
            print(
                "'checkpoint.load()' is deprecated. Please use the new checkpoint API"
            )
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
    def __init__(
        self,
        name: str,
        root_dir: str,
        dtype: Optional[dtype_util.dtype] = None,
        shape: Optional[Sequence[int]] = None,
    ):
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
        assert self.shape is not None
        SLICE_LEN = 8192
        start_idx = 0
        with open(self.file_path_, "rb") as f:
            size = np.prod(self.shape).item()
            while start_idx < size:
                remainder = self.shape[-1]
                while remainder > 0:
                    length = SLICE_LEN if remainder >= SLICE_LEN else remainder
                    remainder -= length
                    stop_idx = start_idx + length
                    np_dtype = np.dtype(
                        dtype_util.convert_oneflow_dtype_to_numpy_dtype(self.dtype)
                    )
                    slice = f.read(length * np_dtype.itemsize)
                    yield start_idx, stop_idx, np.frombuffer(
                        slice, dtype=np_dtype,
                    ).reshape([1] * (len(self.shape) - 1) + [-1])
                    start_idx = stop_idx

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
            raise RuntimeError(
                'The variable "{}" does not have meta info'.format(self.name)
            )
        return np.fromfile(
            self.file_path_,
            dtype=dtype_util.convert_oneflow_dtype_to_numpy_dtype(self.dtype),
        ).reshape(self.shape)


@oneflow_export("get_all_variables")
@session_ctx.try_init_default_session
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
@session_ctx.try_init_default_session
def load(path):
    var_dict = {}
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):
            var_path = os.path.join(path, f, "out")
            if os.path.isfile(var_path):
                var_dict[f] = FileBackendVariableBlob(f, path)
    return var_dict


def read_slice_from_blob(blob, scope_symbol_id):
    assert blob.shape is not None
    SLICE_LEN = 8192
    start_idx = 0
    size = np.prod(blob.shape).item()
    while start_idx < size:
        remainder = blob.shape[-1]
        while remainder > 0:
            length = SLICE_LEN if remainder >= SLICE_LEN else remainder
            remainder -= length
            stop_idx = start_idx + length
            start = np.unravel_index(start_idx, blob.shape)
            stop = np.unravel_index(stop_idx - 1, blob.shape)
            stop = [x + 1 for x in stop]
            yield logical_slice(
                blob.blob_object, start, stop, [1] * len(blob.shape), scope_symbol_id
            )
            start_idx = stop_idx


@oneflow_export("save")
@session_ctx.try_init_default_session
def save(var_dict, path):
    os.makedirs(path, exist_ok=True)
    meta_infos = variable_meta_info_pb.VariableMetaInfos()
    for name, var in var_dict.items():
        meta_info = meta_infos.name2meta_info[name]
        meta_info.shape.dim[:] = var.shape
        meta_info.data_type = var.dtype.oneflow_proto_dtype
        param_path = os.path.join(path, name, "out")
        os.makedirs(os.path.dirname(param_path), exist_ok=True)
        sess = session_ctx.GetDefaultSession()
        var_op_conf = sess.OpConf4InterfaceOpName(name)
        with open(param_path, "wb") as f:
            for slice in read_slice_from_blob(var, var_op_conf.scope_symbol_id):
                f.write(slice.tobytes())
    with open(os.path.join(path, META_INFO_FILENAME), "w") as f:
        f.write(text_format.MessageToString(meta_infos))


def logical_slice(input_blob_object, start, stop, step, scope_symbol_id):
    def AsyncSlice(Yield):
        def build(builder):
            op_conf = op_conf_pb.OperatorConf()
            device_tag = oneflow.current_scope().device_parallel_desc_symbol.device_tag
            op_conf.device_tag = device_tag
            op_conf.name = "logical_slice"
            op_conf.user_conf.op_type_name = "logical_slice"
            op_conf.user_conf.input["x"].s.append("logical_slice/x_0")
            op_conf.user_conf.output["y"].s.append("logical_slice/y_0")
            parallel_conf = input_blob_object.parallel_desc_symbol.parallel_conf
            op_conf.user_conf.attr["parallel_conf"].at_string = str(parallel_conf)
            attribute = user_op_attr_util.UserOpAttrVal()
            attribute.at_list_int64.val[:] = start
            op_conf.user_conf.attr["start"].CopyFrom(attribute)
            attribute = user_op_attr_util.UserOpAttrVal()
            attribute.at_list_int64.val[:] = stop
            op_conf.user_conf.attr["stop"].CopyFrom(attribute)
            attribute = user_op_attr_util.UserOpAttrVal()
            attribute.at_list_int64.val[:] = step
            op_conf.user_conf.attr["step"].CopyFrom(attribute)
            bn_in_op2blob_object = dict(x_0=input_blob_object)
            op_attribute = op_infer_util.Infer(
                op_conf, bn_in_op2blob_object, scope_symbol_id
            )
            builder.StatelessCall(
                op_attribute,
                parallel_conf=parallel_conf,
                bn_in_op2blob_object=bn_in_op2blob_object,
            )
            Yield(bn_in_op2blob_object["y_0"])

        vm_util.LogicalRun(build)

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = "system_checkpoint"
    lbi.blob_name = "system_checkpoint"

    blob_object = async_util.Await(1, AsyncSlice)[0]

    blob = remote_blob_util.EagerConsistentBlob(
        lbi, blob_object=blob_object, job_name="system_checkpoint"
    )
    return blob.numpy()


def get_variable_blob_from_numpy(np_array: np.ndarray):
    flow_dtype = dtype_util.convert_numpy_dtype_to_oneflow_dtype(np_array.dtype)
    op_conf = get_variable.GenerateVariableOpConf(
        name="system_checkpoint",
        shape=np_array.shape,
        dtype=flow_dtype,
        initializer=initializer_util.zeros_initializer(dtype=flow_dtype),
        trainable=False,
        need_root_path=False,
    )
    device_tag = oneflow.current_scope().device_parallel_desc_symbol.device_tag
    op_conf.device_tag = device_tag
    bn_in_op2blob_object = {}
    op_attribute = op_infer_util.Infer(op_conf, bn_in_op2blob_object)
    bn_in_op2blob_object = {}

    def BuildInstruction(builder):
        parallel_conf = (
            oneflow.current_scope().device_parallel_desc_symbol.parallel_conf
        )
        builder.StatelessCall(
            op_attribute, parallel_conf, bn_in_op2blob_object=bn_in_op2blob_object
        )

    vm_util.LogicalRun(BuildInstruction)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_attribute.op_conf.name
    lbi.blob_name = op_attribute.op_conf.variable_conf.out
    var_blob = remote_blob_util.EagerConsistentBlob(
        lbi, job_name="system_checkpoint", blob_object=bn_in_op2blob_object["out"]
    )
    interface_op_read_and_write.FeedValueToInterfaceBlobObject(
        var_blob.blob_object, np_array
    )
    return var_blob


def slice_assign(
    ref_blob_object, value_blob_object, start, stop, step, scope_symbol_id
):
    def BuildAssignInstruction(builder):
        op_conf = op_conf_pb.OperatorConf()
        device_tag = oneflow.current_scope().device_parallel_desc_symbol.device_tag
        op_conf.device_tag = device_tag
        op_conf.name = "slice_assign"
        op_conf.user_conf.op_type_name = "slice_assign"
        op_conf.user_conf.input["value"].s.append("slice_assign/value_0")
        op_conf.user_conf.input["ref"].s.append("slice_assign/ref_0")
        parallel_conf = ref_blob_object.parallel_desc_symbol.parallel_conf
        op_conf.user_conf.attr["parallel_conf"].at_string = str(parallel_conf)
        attribute = user_op_attr_util.UserOpAttrVal()
        attribute.at_list_int64.val[:] = start
        op_conf.user_conf.attr["start"].CopyFrom(attribute)
        attribute = user_op_attr_util.UserOpAttrVal()
        attribute.at_list_int64.val[:] = stop
        op_conf.user_conf.attr["stop"].CopyFrom(attribute)
        attribute = user_op_attr_util.UserOpAttrVal()
        attribute.at_list_int64.val[:] = step
        op_conf.user_conf.attr["step"].CopyFrom(attribute)
        bn_in_op2blob_object = dict(ref_0=ref_blob_object, value_0=value_blob_object)
        op_attribute = op_infer_util.Infer(
            op_conf, bn_in_op2blob_object, scope_symbol_id
        )
        builder.StatelessCall(
            op_attribute,
            parallel_conf=parallel_conf,
            bn_in_op2blob_object=bn_in_op2blob_object,
        )

    vm_util.LogicalRun(BuildAssignInstruction)
    blob_cache_util.TryDisableBlobCache(ref_blob_object)


def _FeedValueToVariable(var_name, value_blob):
    if isinstance(value_blob, EagerBlobTrait):
        var_blob = interface_op_read_and_write.GetEagerInterfaceBlob(var_name)
        interface_op_read_and_write.Assign(value_blob, var_blob)
    elif isinstance(value_blob, FileBackendVariableBlob):
        var_blob = interface_op_read_and_write.GetEagerInterfaceBlob(var_name)
        if not value_blob.has_meta_info_:
            value_blob = FileBackendVariableBlob(
                value_blob.name, value_blob.root_dir_, var_blob.dtype, var_blob.shape
            )
        assert var_blob.shape == value_blob.shape
        sess = session_ctx.GetDefaultSession()
        var_op_conf = sess.OpConf4InterfaceOpName(var_name)
        scope_symbol_id = var_op_conf.scope_symbol_id
        for start_idx, stop_idx, slice in value_blob.read_slice_as_numpy():
            start_nd_idx = np.unravel_index(start_idx, value_blob.shape)
            stop_idx -= 1
            stop_nd_idx = np.unravel_index(stop_idx, value_blob.shape)
            stop_nd_idx = [x + 1 for x in stop_nd_idx]
            value_eager_blob = get_variable_blob_from_numpy(slice)
            slice_assign(
                var_blob.blob_object,
                value_eager_blob.blob_object,
                start_nd_idx,
                stop_nd_idx,
                [1] * len(value_blob.shape),
                scope_symbol_id,
            )
    else:
        raise RuntimeError("Unknown value_blob type: " + type(value_blob).__name__)


@oneflow_export("checkpoint.load_variables")
@session_ctx.try_init_default_session
def load_variables(var_dict, ignore_mismatch=False):
    for name, var in var_dict.items():
        if name in get_all_variables():
            _FeedValueToVariable(name, var)
        else:
            if not ignore_mismatch:
                raise RuntimeError('"{}" is not a variable name'.format(name))


def init():
    SLICE_LEN = 8192
    sess = session_ctx.GetDefaultSession()
    for op_name, var_blob in get_all_variables().items():
        rng = np.random.default_rng()
        var_op_conf = sess.OpConf4InterfaceOpName(op_name)
        scope_symbol_id = var_op_conf.scope_symbol_id
        start_idx = 0
        size = np.prod(var_blob.shape).item()
        while start_idx < size:
            remainder = var_blob.shape[-1]
            while remainder > 0:
                length = SLICE_LEN if remainder >= SLICE_LEN else remainder
                remainder -= length
                stop_idx = start_idx + length
                start_nd_idx = np.unravel_index(start_idx, var_blob.shape)
                stop_nd_idx = np.unravel_index(stop_idx - 1, var_blob.shape)
                stop_nd_idx = [x + 1 for x in stop_nd_idx]
                np_dtype = np.dtype(
                    dtype_util.convert_oneflow_dtype_to_numpy_dtype(var_blob.dtype)
                )
                vals = []
                for _ in range(length):
                    # TODO: dtype
                    vals.append(rng.normal())
                vals = (
                    np.array(vals)
                    .astype(np_dtype)
                    .reshape([1] * (len(var_blob.shape) - 1) + [-1])
                )

                value_eager_blob = get_variable_blob_from_numpy(vals)
                slice_assign(
                    var_blob.blob_object,
                    value_eager_blob.blob_object,
                    start_nd_idx,
                    stop_nd_idx,
                    [1] * len(var_blob.shape),
                    scope_symbol_id,
                )
                start_idx = stop_idx
