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
import oneflow_api
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.ops.initializer_util as initializer_util
import oneflow.python.framework.id_util as id_util
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
import oneflow.core.framework.user_op_attr_pb2 as attr_value_pb
from oneflow.python.experimental import interface_op_read_and_write
from oneflow.python.framework.remote_blob import EagerBlobTrait
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.ops.get_variable as get_variable

from oneflow.python.oneflow_export import oneflow_export
from typing import Dict, List, Union, Sequence, Optional


META_INFO_FILENAME = "meta"
DATA_FILENAME = "out"


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

    def list_checkpoints(self) -> List[str]:
        def is_snapshot(name):
            return name.startswith(self._prefix)

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
            LoadVariables(Load(self._GetSnapshotPath(name)))
        else:
            self.save()

    def save(self) -> None:
        Save(GetAllVariables(), self._GetSnapshotPath(self._NextSnapshotName()))

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
                if f == DATA_FILENAME and os.path.isfile(fpath):
                    has_out_subfile = True

            if not has_out_subfile:
                continue

            assert file not in self.name2path_
            self.name2path_[file] = os.path.join(file_path, DATA_FILENAME)

    def get_snapshot_path(self, name):
        try:
            return self.name2path_[name]
        except KeyError:
            return None


class FileBackendVariableBlob:
    def __init__(
        self,
        var_dir: str,
        dtype: Optional[dtype_util.dtype] = None,
        shape: Optional[Sequence[int]] = None,
    ):
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
            else:
                pass

    def GetSlicesAsNumpy(self):
        assert self.shape is not None
        np_dtype = np.dtype(dtype_util.convert_oneflow_dtype_to_numpy_dtype(self.dtype))
        with open(self.file_path_, "rb") as f:

            def ReadFromFile(blob, start_nd_idx, stop_nd_idx, length):
                slice = f.read(length * np_dtype.itemsize)
                return np.frombuffer(slice, dtype=np_dtype,).reshape(
                    np.array(stop_nd_idx) - np.array(start_nd_idx)
                )

            yield from _ForEverySlice(self, ReadFromFile)

    @property
    def file_path_(self):
        return os.path.join(self.var_dir_, DATA_FILENAME)

    @property
    def shape(self):
        return self.shape_

    @property
    def quant_info(self):
        raise NotImplementedError()

    @property
    def dtype(self):
        return self.dtype_

    def numpy(self) -> np.ndarray:
        if not self.has_meta_info_:
            raise RuntimeError("This variable does not have meta info")
        return np.fromfile(
            self.file_path_,
            dtype=dtype_util.convert_oneflow_dtype_to_numpy_dtype(self.dtype),
        ).reshape(self.shape)


@oneflow_export("get_all_variables")
@session_ctx.try_init_default_session
def GetAllVariables() -> Dict[str, remote_blob_util.EagerConsistentBlob]:
    sess = session_ctx.GetDefaultSession()
    interface_ops = sess.interface_ops
    variables = {}
    for op in interface_ops:
        op_attr = sess.OpAttribute4InterfaceOpName(op)
        if op_attr.op_conf.WhichOneof("op_type") != "variable_conf":
            continue
        variables[op] = interface_op_read_and_write.GetEagerInterfaceBlob(op)
    return variables


def _LoadSingleVariable(path):
    if os.path.isfile(os.path.join(path, DATA_FILENAME)):
        return FileBackendVariableBlob(path)
    return None


@oneflow_export("load")
@session_ctx.try_init_default_session
def Load(path):
    assert os.path.isdir(path), "Directory {} doesn't exist!".format(path)
    single_var = _LoadSingleVariable(path)
    if single_var is not None:
        return single_var
    var_dict = {}
    for f in os.listdir(path):
        var_dir = os.path.join(path, f)
        var = _LoadSingleVariable(var_dir)
        if var is not None:
            var_dict[f] = var
    return var_dict


def _GetScopeSymbolIdFromEagerBlob(blob):
    name = blob.logical_blob_name.split("/")[0]
    sess = session_ctx.GetDefaultSession()
    op_conf = sess.OpConf4InterfaceOpName(name)
    scope_symbol_id = op_conf.scope_symbol_id
    return scope_symbol_id


def _ReadSliceFromEagerBlob(blob):
    def ReadSlice(blob, start_nd_idx, stop_nd_idx, length):
        return _LogicalSlice(blob, start_nd_idx, stop_nd_idx)

    yield from _ForEverySlice(blob, ReadSlice)


@oneflow_export("save")
@session_ctx.try_init_default_session
def Save(var_dict, path):
    def IsFileOrNonEmptyDir(path):
        if os.path.isfile(path):
            return True
        if os.path.isdir(path) and len(os.listdir(path)) != 0:
            return True
        return False

    assert not IsFileOrNonEmptyDir(
        path
    ), "Non-empty directory {} already exists!".format(path)
    os.makedirs(path, exist_ok=True)
    for name, var in var_dict.items():
        meta_info = variable_meta_info_pb.VariableMetaInfo()
        meta_info.shape.dim[:] = var.shape
        meta_info.data_type = var.dtype.oneflow_proto_dtype
        var_dir = os.path.join(path, name)
        param_path = os.path.join(var_dir, DATA_FILENAME)
        os.makedirs(os.path.dirname(param_path))
        sess = session_ctx.GetDefaultSession()
        var_op_conf = sess.OpConf4InterfaceOpName(name)
        with open(param_path, "wb") as f:
            for _, _, slice in _ReadSliceFromEagerBlob(var):
                f.write(slice.tobytes())
        with open(os.path.join(var_dir, META_INFO_FILENAME), "w") as f:
            f.write(text_format.MessageToString(meta_info))


def _LogicalSlice(input_blob, start, stop):
    op_name = id_util.UniqueStr("system_checkpoint")

    def AsyncSlice(Yield):
        def build(builder):
            op_conf = op_conf_pb.OperatorConf()
            # device_tag doesn't matter for logical_slice op
            device_tag = oneflow.current_scope().device_parallel_desc_symbol.device_tag
            op_conf.device_tag = device_tag
            op_conf.name = op_name
            op_conf.user_conf.op_type_name = "logical_slice"
            op_conf.user_conf.input["x"].s.append("{}/x_0".format(op_name))
            op_conf.user_conf.output["y"].s.append("{}/y_0".format(op_name))
            input_blob_object = input_blob.blob_object
            parallel_conf = input_blob_object.parallel_desc_symbol.parallel_conf
            op_conf.user_conf.attr["parallel_conf"].at_string = str(parallel_conf)
            attribute = attr_value_pb.AttrValue()
            attribute.at_list_int64.val[:] = start
            op_conf.user_conf.attr["start"].CopyFrom(attribute)
            attribute = attr_value_pb.AttrValue()
            attribute.at_list_int64.val[:] = stop
            op_conf.user_conf.attr["stop"].CopyFrom(attribute)
            attribute = attr_value_pb.AttrValue()
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
            Yield(bn_in_op2blob_object["y_0"])

        vm_util.LogicalRun(build)

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_name
    lbi.blob_name = op_name

    blob_object = async_util.Await(1, AsyncSlice)[0]

    blob = remote_blob_util.EagerConsistentBlob(
        lbi, blob_object=blob_object, job_name="system_checkpoint"
    )
    return blob.numpy()


def _GetCpu0VariableBlobFromNumpy(np_array: np.ndarray, dtype: dtype_util.dtype):
    # Note: dtype argument cannot be replaced with 
    # convert_numpy_dtype_to_oneflow_dtype(np_array.dtype), 
    # because np.int8 == np.char and 
    # numpy_dtype_to_oneflow_dtype(oneflow_dtype_to_numpy_dtype(flow.int8)) 
    # may be flow.char
    with oneflow.scope.placement("cpu", "0:0"):
        op_name = id_util.UniqueStr("system_checkpoint")
        op_conf = get_variable.GenerateVariableOpConf(
            name=op_name,
            shape=np_array.shape,
            dtype=dtype,
            initializer=initializer_util.zeros_initializer(dtype=dtype),
            trainable=False,
            need_root_path=False,
        )
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
        # device_tag doesn't matter for logical_slice_assign op
        device_tag = oneflow.current_scope().device_parallel_desc_symbol.device_tag
        op_conf.device_tag = device_tag
        op_name = id_util.UniqueStr("system_checkpoint")
        op_conf.name = op_name
        op_conf.user_conf.op_type_name = "logical_slice_assign"
        op_conf.user_conf.input["value"].s.append("{}/value_0".format(op_name))
        op_conf.user_conf.input["ref"].s.append("{}/ref_0".format(op_name))
        parallel_conf = ref_blob_object.parallel_desc_symbol.parallel_conf
        op_conf.user_conf.attr["parallel_conf"].at_string = str(parallel_conf)
        attribute = attr_value_pb.AttrValue()
        attribute.at_list_int64.val[:] = start
        op_conf.user_conf.attr["start"].CopyFrom(attribute)
        attribute = attr_value_pb.AttrValue()
        attribute.at_list_int64.val[:] = stop
        op_conf.user_conf.attr["stop"].CopyFrom(attribute)
        attribute = attr_value_pb.AttrValue()
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
            slice_value_blob = _GetCpu0VariableBlobFromNumpy(slice, var_blob.dype)
            _LogicalSliceAssign(
                var_blob, slice_value_blob, start, stop,
            )
    elif isinstance(value_blob, FileBackendVariableBlob):
        if not value_blob.has_meta_info_:
            value_blob = FileBackendVariableBlob(
                value_blob.var_dir_, var_blob.dtype, var_blob.shape
            )
        assert var_blob.shape == value_blob.shape, "{} vs {}".format(
            var_blob.shape, value_blob.shape
        )
        for start, stop, slice in value_blob.GetSlicesAsNumpy():
            slice_value_blob = _GetCpu0VariableBlobFromNumpy(slice, var_blob.dype)
            _LogicalSliceAssign(
                var_blob, slice_value_blob, start, stop,
            )
    else:
        raise RuntimeError("Unknown value_blob type: " + type(value_blob).__name__)


@oneflow_export("load_variables")
@session_ctx.try_init_default_session
def LoadVariables(var_dict, ignore_mismatch=False):
    for name, var in var_dict.items():
        if name in GetAllVariables():
            _FeedValueToVariable(name, var)
        else:
            if not ignore_mismatch:
                raise RuntimeError('"{}" is not a variable name'.format(name))
    oneflow_api.eager.Sync()


def _ForEverySlice(var_blob, f):
    # For current implementation (transport data by grpc), SLICE_BYTES must be lower than 64M
    SLICE_BYTES = 20 * 1024 * 1024
    np_dtype = np.dtype(dtype_util.convert_oneflow_dtype_to_numpy_dtype(var_blob.dtype))
    SLICE_LEN = SLICE_BYTES // np_dtype.itemsize
    start_idx = 0
    size = np.prod(var_blob.shape).astype(np.int).item()
    cnt = 1
    for axis in reversed(range(len(var_blob.shape))):
        cnt *= var_blob.shape[axis]
        if cnt > SLICE_LEN:
            break
    unit_size = np.prod(var_blob.shape[axis + 1 :]).astype(np.int).item()
    max_unit_num = SLICE_LEN // unit_size
    while start_idx < size:
        remainder = var_blob.shape[axis]
        while remainder > 0:
            unit_num = max_unit_num if remainder >= max_unit_num else remainder
            length = unit_num * unit_size
            remainder -= unit_num
            stop_idx = start_idx + length
            start_nd_idx = np.unravel_index(start_idx, var_blob.shape)
            stop_nd_idx = np.unravel_index(stop_idx - 1, var_blob.shape)
            stop_nd_idx = tuple([x + 1 for x in stop_nd_idx])
            yield start_nd_idx, stop_nd_idx, f(
                var_blob, start_nd_idx, stop_nd_idx, length
            )
            start_idx = stop_idx


def Init():
    sess = session_ctx.GetDefaultSession()
    for op_name, var_blob in GetAllVariables().items():
        var_conf = sess.OpConf4InterfaceOpName(op_name).variable_conf
        if var_conf.HasField("initialize_with_snapshot"):
            initialize_with_snapshot_conf = var_conf.initialize_with_snapshot
            var_dir = os.path.dirname(
                os.path.join(
                    initialize_with_snapshot_conf.path,
                    initialize_with_snapshot_conf.key,
                )
            )
            LoadVariables({op_name: Load(var_dir)})
            continue
        g = initializer_util.GetInitializerGenerator(
            var_conf.initializer, var_conf.random_seed
        )

        def GenerateValueAndAssign(var_blob, start_nd_idx, stop_nd_idx, length):
            np_dtype = np.dtype(
                dtype_util.convert_oneflow_dtype_to_numpy_dtype(var_blob.dtype)
            )
            vals = g(length)
            vals = (
                np.array(vals)
                .astype(np_dtype)
                .reshape(np.array(stop_nd_idx) - np.array(start_nd_idx))
            )

            slice_value_blob = _GetCpu0VariableBlobFromNumpy(vals, var_blob.dtype)
            _LogicalSliceAssign(
                var_blob, slice_value_blob, start_nd_idx, stop_nd_idx,
            )

        # we don't care about the return value,
        # only want to run f on every slice
        for _ in _ForEverySlice(var_blob, GenerateValueAndAssign):
            pass
    oneflow_api.eager.Sync()
