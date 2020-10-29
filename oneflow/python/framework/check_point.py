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
import oneflow.python.framework.op_arg_util as op_arg_util
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
        var_dir: str,
        dtype: Optional[dtype_util.dtype] = None,
        shape: Optional[Sequence[int]] = None,
    ):
        self.name_ = name
        self.var_dir_ = var_dir
        meta_info_path = os.path.join(self.var_dir_, META_INFO_FILENAME)
        if os.path.exists(meta_info_path):
            meta_info = variable_meta_info_pb.VariableMetaInfo()
            with open(meta_info_path) as f:
                text_format.Parse(f.read(), meta_info)
            self.has_meta_info_ = True
        else:
            self.has_meta_info_ = False

        if self.has_meta_info_:
            assert dtype is None and shape is None
            self.shape_ = tuple(meta_info.shape.dim)
            self.dtype_ = dtype_util.convert_proto_dtype_to_oneflow_dtype(
                meta_info.data_type
            )
        else:
            if shape is not None and dtype is not None:
                self.shape_ = shape
                self.dtype_ = dtype
                self.has_meta_info_ = True
            elif shape is not None or dtype is not None:
                raise RuntimeError("both or neither of shape and dtype should be None")

    def GetSlicesAsNumpy(self):
        assert self.shape is not None
        np_dtype = np.dtype(dtype_util.convert_oneflow_dtype_to_numpy_dtype(self.dtype))
        with open(self.file_path_, "rb") as f:

            def ReadFromFile(blob, start_nd_idx, stop_nd_idx, length):
                slice = f.read(length * np_dtype.itemsize)
                return np.frombuffer(slice, dtype=np_dtype,).reshape(
                    [1] * (len(self.shape) - 1) + [-1]
                )

            yield from _ForEverySlice(self, ReadFromFile)

    @property
    def file_path_(self):
        return os.path.join(self.var_dir_, "out")

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
        var_dir = os.path.join(path, f)
        if os.path.isdir(var_dir):
            var_path = os.path.join(var_dir, "out")
            if os.path.isfile(var_path):
                var_dict[f] = FileBackendVariableBlob(f, var_dir)
    return var_dict


def _GetScopeSymbolIdFromEagerBlob(blob):
    name = blob.logical_blob_name.split("/")[0]
    sess = session_ctx.GetDefaultSession()
    op_conf = sess.OpConf4InterfaceOpName(name)
    scope_symbol_id = op_conf.scope_symbol_id
    return scope_symbol_id


def _ReadSliceFromEagerBlob(blob):
    def ReadSlice(blob, start_nd_idx, stop_nd_idx, length):
        return _LogicalSlice(blob, start_nd_idx, stop_nd_idx,)

    yield from _ForEverySlice(blob, ReadSlice)


@oneflow_export("save")
@session_ctx.try_init_default_session
def save(var_dict, path):
    os.makedirs(path, exist_ok=True)
    for name, var in var_dict.items():
        meta_info = variable_meta_info_pb.VariableMetaInfo()
        meta_info.shape.dim[:] = var.shape
        meta_info.data_type = var.dtype.oneflow_proto_dtype
        var_dir = os.path.join(path, name)
        param_path = os.path.join(var_dir, "out")
        os.makedirs(os.path.dirname(param_path), exist_ok=True)
        sess = session_ctx.GetDefaultSession()
        var_op_conf = sess.OpConf4InterfaceOpName(name)
        with open(param_path, "wb") as f:
            for _, _, slice in _ReadSliceFromEagerBlob(var):
                f.write(slice.tobytes())
        with open(os.path.join(var_dir, META_INFO_FILENAME), "w") as f:
            f.write(text_format.MessageToString(meta_info))


def _LogicalSlice(input_blob, start, stop):
    def AsyncSlice(Yield):
        def build(builder):
            op_conf = op_conf_pb.OperatorConf()
            device_tag = oneflow.current_scope().device_parallel_desc_symbol.device_tag
            op_conf.device_tag = device_tag
            op_conf.name = "logical_slice"
            op_conf.user_conf.op_type_name = "logical_slice"
            op_conf.user_conf.input["x"].s.append("logical_slice/x_0")
            op_conf.user_conf.output["y"].s.append("logical_slice/y_0")
            input_blob_object = input_blob.blob_object
            parallel_conf = input_blob_object.parallel_desc_symbol.parallel_conf
            op_conf.user_conf.attr["parallel_conf"].at_string = str(parallel_conf)
            attribute = user_op_attr_util.UserOpAttrVal()
            attribute.at_list_int64.val[:] = start
            op_conf.user_conf.attr["start"].CopyFrom(attribute)
            attribute = user_op_attr_util.UserOpAttrVal()
            attribute.at_list_int64.val[:] = stop
            op_conf.user_conf.attr["stop"].CopyFrom(attribute)
            attribute = user_op_attr_util.UserOpAttrVal()
            step = [1] * len(start)
            attribute.at_list_int64.val[:] = step
            op_conf.user_conf.attr["step"].CopyFrom(attribute)
            bn_in_op2blob_object = dict(x_0=input_blob_object)
            scope_symbol_id = _GetScopeSymbolIdFromEagerBlob(input_blob)
            op_attribute = op_infer_util.Infer(
                op_conf, bn_in_op2blob_object, scope_symbol_id
            )
            builder.StatelessCall(
                op_attribute,
                parallel_conf=parallel_conf,
                bn_in_op2blob_object=bn_in_op2blob_object,
            )
            parallel_desc_symbol = oneflow.current_scope().device_parallel_desc_symbol
            op_arg_parallel_attr = op_arg_util.MakeBroadcastOpArgParallelAttribute(
                parallel_desc_symbol
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


def _GetVariableBlobFromNumpy(np_array: np.ndarray):
    with oneflow.scope.placement("cpu", "0:0"):
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


def _LogicalSliceAssign(ref_blob, value_blob, start, stop):
    ref_blob_object = ref_blob.blob_object
    value_blob_object = value_blob.blob_object

    def BuildAssignInstruction(builder):
        op_conf = op_conf_pb.OperatorConf()
        device_tag = oneflow.current_scope().device_parallel_desc_symbol.device_tag
        op_conf.device_tag = device_tag
        op_conf.name = "logical_slice_assign"
        op_conf.user_conf.op_type_name = "logical_slice_assign"
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
        step = [1] * len(start)
        attribute.at_list_int64.val[:] = step
        op_conf.user_conf.attr["step"].CopyFrom(attribute)
        bn_in_op2blob_object = dict(ref_0=ref_blob_object, value_0=value_blob_object)
        scope_symbol_id = _GetScopeSymbolIdFromEagerBlob(ref_blob)
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
    sess = session_ctx.GetDefaultSession()
    var_op_conf = sess.OpConf4InterfaceOpName(var_name)
    var_blob = interface_op_read_and_write.GetEagerInterfaceBlob(var_name)
    if isinstance(value_blob, EagerBlobTrait):
        value_name = value_blob.logical_blob_name.split("/")[0]
        value_op_conf = sess.OpConf4InterfaceOpName(value_name)
        for start, stop, slice in _ReadSliceFromEagerBlob(value_blob):
            slice_value_blob = _GetVariableBlobFromNumpy(slice)
            _LogicalSliceAssign(
                var_blob, slice_value_blob, start, stop,
            )
    elif isinstance(value_blob, FileBackendVariableBlob):
        if not value_blob.has_meta_info_:
            value_blob = FileBackendVariableBlob(
                value_blob.name, value_blob.var_dir_, var_blob.dtype, var_blob.shape
            )
        assert var_blob.shape == value_blob.shape
        for start, stop, slice in value_blob.GetSlicesAsNumpy():
            slice_value_blob = _GetVariableBlobFromNumpy(slice)
            _LogicalSliceAssign(
                var_blob, slice_value_blob, start, stop,
            )
    else:
        raise RuntimeError("Unknown value_blob type: " + type(value_blob).__name__)


@oneflow_export("load_variables")
@session_ctx.try_init_default_session
def load_variables(var_dict, ignore_mismatch=False):
    for name, var in var_dict.items():
        if name in get_all_variables():
            _FeedValueToVariable(name, var)
        else:
            if not ignore_mismatch:
                raise RuntimeError('"{}" is not a variable name'.format(name))


def _ForEverySlice(var_blob, f):
    SLICE_LEN = 8192
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
            yield start_nd_idx, stop_nd_idx, f(
                var_blob, start_nd_idx, stop_nd_idx, length
            )
            start_idx = stop_idx


_init_map = {}


def register_initializer(flow_initializer):
    def deco(func):
        _init_map[flow_initializer] = func
        return func

    return deco


def _GetInitializerGenerator(initializer_conf):
    f = None
    for m in _init_map:
        if initializer_conf.HasField(m):
            f = _init_map[m]
            break
    assert f is not None, initializer_conf
    yield from f(getattr(initializer_conf, m))


@register_initializer("constant_conf")
@register_initializer("constant_int_conf")
def constant_initializer(
    initializer_conf: Union[
        op_conf_pb.ConstantInitializerConf, op_conf_pb.ConstantIntInitializerConf
    ]
):
    while True:
        yield initializer_conf.value


@register_initializer("random_normal_conf")
def random_normal_initializer(initializer_conf: op_conf_pb.RandomNormalInitializerConf):
    rng = np.random.default_rng()
    while True:
        yield rng.normal(loc=initializer_conf.mean, scale=initializer_conf.std)


def Init():
    sess = session_ctx.GetDefaultSession()
    for op_name, var_blob in get_all_variables().items():
        rng = np.random.default_rng()
        var_op_conf = sess.OpConf4InterfaceOpName(op_name)
        np_dtype = np.dtype(
            dtype_util.convert_oneflow_dtype_to_numpy_dtype(var_blob.dtype)
        )

        g = _GetInitializerGenerator(var_op_conf.variable_conf.initializer)

        def GenerateValueAndAssign(var_blob, start_nd_idx, stop_nd_idx, length):
            vals = []
            for _ in range(length):
                vals.append(next(g))
            vals = (
                np.array(vals)
                .astype(np_dtype)
                .reshape([1] * (len(var_blob.shape) - 1) + [-1])
            )

            slice_value_blob = _GetVariableBlobFromNumpy(vals)
            _LogicalSliceAssign(
                var_blob, slice_value_blob, start_nd_idx, stop_nd_idx,
            )

        # we don't care about the return value,
        # only want to run f on every slice
        for _ in _ForEverySlice(var_blob, GenerateValueAndAssign):
            pass
