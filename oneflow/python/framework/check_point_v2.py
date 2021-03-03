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
import os

import numpy as np
from google.protobuf import text_format

import oneflow
import oneflow_api
import oneflow.python.eager.blob_register as blob_register_util
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.ops.initializer_util as initializer_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.lib.core.async_util as async_util
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.boxing_util as boxing_util
import oneflow.python.eager.op_infer_util as op_infer_util
import oneflow.core.framework.variable_meta_info_pb2 as variable_meta_info_pb
import oneflow.core.framework.user_op_attr_pb2 as attr_value_pb
from oneflow.python.experimental import interface_op_read_and_write
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.ops.get_variable as get_variable

from oneflow.python.oneflow_export import oneflow_export
import oneflow_api.oneflow.core.register.logical_blob_id as lbi_util
from oneflow_api import EagerBlobTrait
import oneflow_api
from typing import Any, Callable, Dict, List, Union, Sequence, Optional, Iterable, Tuple


META_INFO_FILENAME = "meta"
DATA_FILENAME = "out"
FAKE_JOB_NAME = "system_checkpoint"
OP_PREFIX = "system_checkpoint"


blob_register = blob_register_util.GetDefaultBlobRegister()


class FileBackendVariableBlob:
    def __init__(
        self,
        var_dir: str,
        dtype: Optional[oneflow.dtype] = None,
        shape: Optional[Sequence[int]] = None,
    ):
        data_path = os.path.join(var_dir, DATA_FILENAME)
        assert os.path.isfile(data_path)
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

        if self.has_meta_info_:
            itemsize = np.dtype(
                dtype_util.convert_oneflow_dtype_to_numpy_dtype(self.dtype_)
            ).itemsize
            assert os.path.getsize(data_path) == np.prod(self.shape).item() * itemsize

    @property
    def file_path(self) -> str:
        return os.path.join(self.var_dir_, DATA_FILENAME)

    @property
    def shape(self) -> Tuple[int]:
        return self.shape_

    @property
    def quant_info(self):
        raise NotImplementedError()

    @property
    def dtype(self) -> oneflow.dtype:
        return self.dtype_

    def numpy(self) -> np.ndarray:
        if not self.has_meta_info_:
            raise RuntimeError("This variable does not have meta info")
        return np.fromfile(
            self.file_path,
            dtype=dtype_util.convert_oneflow_dtype_to_numpy_dtype(self.dtype),
        ).reshape(self.shape)


ValueContainer = Union[EagerBlobTrait, FileBackendVariableBlob, np.ndarray]


def _ElemCnt(shape):
    return np.prod(shape).astype(np.int).item()


@oneflow_export("get_all_variables")
@session_ctx.try_init_default_session
def GetAllVariables() -> Dict[str, oneflow_api.EagerConsistentBlob]:
    """
    Get all variables of all jobs as a dict.
    """
    oneflow.sync_default_session()

    sess = session_ctx.GetDefaultSession()
    interface_ops = sess.interface_ops
    variables = {}
    for op in interface_ops:
        op_attr = sess.OpAttribute4InterfaceOpName(op)
        if op_attr.op_conf.WhichOneof("op_type") != "variable_conf":
            continue
        variables[op] = interface_op_read_and_write.GetEagerInterfaceBlob(op)
    return variables


def _LoadSingleVariable(path: str) -> Optional[FileBackendVariableBlob]:
    if os.path.isfile(os.path.join(path, DATA_FILENAME)):
        return FileBackendVariableBlob(path)
    return None


@oneflow_export("checkpoint.get")
@session_ctx.try_init_default_session
def GetCheckpoint(
    path: str,
) -> Union[Dict[str, FileBackendVariableBlob], FileBackendVariableBlob]:
    """
    Load variable(s) from file system.
    """
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


def _GetOpNameFromLbn(lbn):
    return lbn.split("/")[0]


def _GetScopeSymbolIdFromEagerBlob(blob):
    name = _GetOpNameFromLbn(blob.logical_blob_name)
    sess = session_ctx.GetDefaultSession()
    op_conf = sess.OpAttribute4InterfaceOpName(name).op_conf
    scope_symbol_id = op_conf.scope_symbol_id
    return scope_symbol_id


def _ReadSlice(
    container: ValueContainer,
) -> Iterable[Tuple[Sequence[int], Sequence[int], np.ndarray]]:
    """
    Return a generator which iterates over the input blob or array and yields
    (start_nd_idx, stop_nd_idx, slice_np_array)
    """
    if isinstance(container, EagerBlobTrait):

        def ReadFromEagerBlob(eager_blob, start_nd_idx, stop_nd_idx):
            return _LogicalSlice(eager_blob, start_nd_idx, stop_nd_idx)

        yield from _ForEachSlice(container, ReadFromEagerBlob)
    elif isinstance(container, FileBackendVariableBlob):
        np_dtype = np.dtype(
            dtype_util.convert_oneflow_dtype_to_numpy_dtype(container.dtype)
        )
        with open(container.file_path, "rb") as f:

            def ReadFromFile(_, start_nd_idx, stop_nd_idx):
                length = _ElemCnt(np.array(stop_nd_idx) - np.array(start_nd_idx))
                slice = f.read(length * np_dtype.itemsize)
                return np.frombuffer(slice, dtype=np_dtype,).reshape(
                    np.array(stop_nd_idx) - np.array(start_nd_idx)
                )

            yield from _ForEachSlice(container, ReadFromFile)
    elif isinstance(container, np.ndarray):

        def ReadFromNpArray(array, start_nd_idx, stop_nd_idx):
            slice_objs = []
            for start, stop in zip(start_nd_idx, stop_nd_idx):
                slice_objs.append(slice(start, stop))
            return array[tuple(slice_objs)]

        yield from _ForEachSlice(container, ReadFromNpArray)
    else:
        raise RuntimeError("Unknown type: {}".format(type(container).__name__))


@oneflow_export("checkpoint.save")
@session_ctx.try_init_default_session
def SaveVarDict(
    path: str,
    var_dict: Optional[
        Dict[str, Union[FileBackendVariableBlob, EagerBlobTrait]]
    ] = None,
) -> None:
    """
    Save `var_dict` to `path`
    """
    oneflow.sync_default_session()

    if var_dict is None:
        var_dict = GetAllVariables()

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
        meta_info.data_type = oneflow_api.deprecated.GetProtoDtype4OfDtype(var.dtype)
        var_dir = os.path.join(path, name)
        param_path = os.path.join(var_dir, DATA_FILENAME)
        os.makedirs(os.path.dirname(param_path))
        with open(param_path, "wb") as f:
            for _, _, slice in _ReadSlice(var):
                f.write(slice.tobytes())
        with open(os.path.join(var_dir, META_INFO_FILENAME), "w") as f:
            f.write(text_format.MessageToString(meta_info))
    # write a empty file 'snapshot_done', indicating that
    # the save process finishes normally
    with open(os.path.join(path, "snapshot_done"), "w"):
        pass


def _LogicalSlice(
    input_blob: EagerBlobTrait, start: Sequence[int], stop: Sequence[int]
) -> np.ndarray:
    """
    Construct a logical_slice op and run it by oneflow eager,
    return the sliced result as a numpy ndarray
    """
    op_name = id_util.UniqueStr(OP_PREFIX)

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
            op_conf.user_conf.attr["start"].at_list_int64.val[:] = start
            op_conf.user_conf.attr["stop"].at_list_int64.val[:] = stop
            op_conf.user_conf.attr["step"].at_list_int64.val[:] = [1] * len(start)
            bn_in_op2blob_object = oneflow_api.deprecated.BnInOp2BlobObject()
            bn_in_op2blob_object["x_0"] = input_blob_object
            scope_symbol_id = _GetScopeSymbolIdFromEagerBlob(input_blob)
            op_attribute = op_infer_util.Infer(
                op_conf, bn_in_op2blob_object, scope_symbol_id
            )
            cfg_op_attribute = oneflow_api.deprecated.MakeOpAttributeByString(
                str(op_attribute)
            )
            builder.StatelessCall(
                cfg_op_attribute,
                parallel_conf,
                bn_in_op2blob_object,
                boxing_util.BoxingTo,
                vm_util._FindOrCreateDelegateBlobObject,
            )
            Yield(bn_in_op2blob_object["y_0"])

        vm_util.LogicalRun(build)

    lbi = lbi_util.LogicalBlobId()
    lbi.set_op_name(op_name)
    lbi.set_blob_name(op_name)

    blob_object = async_util.Await(1, AsyncSlice)[0]

    blob = oneflow_api.EagerConsistentBlob(
        lbi,
        blob_object=blob_object,
        blob_register=blob_register,
        job_name=FAKE_JOB_NAME,
    )
    return blob.numpy()


def _GetCpu0VariableBlobFromNumpy(
    np_array: np.ndarray, dtype: oneflow.dtype
) -> oneflow_api.EagerConsistentBlob:
    """
    Add a variable on cpu 0, and feed the value of `np_array`

    Note: dtype argument cannot be eliminated by
    convert_numpy_dtype_to_oneflow_dtype(np_array.dtype),
    because np.int8 == np.char and
    numpy_dtype_to_oneflow_dtype(oneflow_dtype_to_numpy_dtype(flow.int8))
    may be flow.char
    """
    with oneflow.scope.placement("cpu", "0:0"):
        op_name = id_util.UniqueStr(OP_PREFIX)
        op_conf = get_variable.GenerateVariableOpConf(
            name=op_name,
            shape=np_array.shape,
            dtype=dtype,
            initializer=initializer_util.zeros_initializer(dtype=dtype),
            trainable=False,
        )
        current_parallel_desc_sym = oneflow.current_scope().device_parallel_desc_symbol
        device_tag = current_parallel_desc_sym.device_tag
        op_conf.device_tag = device_tag
        op_attribute = op_infer_util.Infer(op_conf, {})
        var_blob = get_variable.CreateEagerVariableBlob(
            op_attribute, job_name=FAKE_JOB_NAME
        )

        interface_op_read_and_write.FeedValueToInterfaceBlobObject(
            var_blob.blob_object, np_array
        )
        return var_blob


def _LogicalSliceAssign(
    ref_blob: EagerBlobTrait,
    value_blob: EagerBlobTrait,
    start: Sequence[int],
    stop: Sequence[int],
) -> None:
    """
    Construct a logical_slice_assign op and run it by oneflow eager
    """
    ref_blob_object = ref_blob.blob_object
    value_blob_object = value_blob.blob_object

    def BuildAssignInstruction(builder):
        op_conf = op_conf_pb.OperatorConf()
        # device_tag doesn't matter for logical_slice_assign op
        device_tag = oneflow.current_scope().device_parallel_desc_symbol.device_tag
        op_conf.device_tag = device_tag
        op_name = id_util.UniqueStr(OP_PREFIX)
        op_conf.name = op_name
        op_conf.user_conf.op_type_name = "logical_slice_assign"
        op_conf.user_conf.input["value"].s.append("{}/value_0".format(op_name))
        op_conf.user_conf.input["ref"].s.append("{}/ref_0".format(op_name))
        parallel_conf = ref_blob_object.parallel_desc_symbol.parallel_conf
        op_conf.user_conf.attr["parallel_conf"].at_string = str(parallel_conf)
        op_conf.user_conf.attr["start"].at_list_int64.val[:] = start
        op_conf.user_conf.attr["stop"].at_list_int64.val[:] = stop
        op_conf.user_conf.attr["step"].at_list_int64.val[:] = [1] * len(start)
        bn_in_op2blob_object = oneflow_api.deprecated.BnInOp2BlobObject()
        bn_in_op2blob_object["ref_0"] = ref_blob_object
        bn_in_op2blob_object["value_0"] = value_blob_object
        scope_symbol_id = _GetScopeSymbolIdFromEagerBlob(ref_blob)
        op_attribute = op_infer_util.Infer(
            op_conf, bn_in_op2blob_object, scope_symbol_id
        )
        cfg_op_attribute = oneflow_api.deprecated.MakeOpAttributeByString(
            str(op_attribute)
        )
        builder.StatelessCall(
            cfg_op_attribute,
            parallel_conf,
            bn_in_op2blob_object,
            boxing_util.BoxingTo,
            vm_util._FindOrCreateDelegateBlobObject,
        )

    vm_util.LogicalRun(BuildAssignInstruction)
    blob_cache_util.TryDisableBlobCache(ref_blob_object)


def _FeedValueToVariable(
    var_blob: oneflow_api.EagerConsistentBlob, value: ValueContainer
) -> None:
    """
    Feed the value of `value` to the variable `var_blob`
    """
    assert isinstance(
        value, (EagerBlobTrait, FileBackendVariableBlob, np.ndarray)
    ), "Unknown value type: {}".format(type(value).__name__)

    if isinstance(value, FileBackendVariableBlob):
        if not value.has_meta_info_:
            value = FileBackendVariableBlob(
                value.var_dir_, var_blob.dtype, var_blob.shape
            )
    assert var_blob.shape == value.shape, "{} vs {}".format(var_blob.shape, value.shape)
    if isinstance(value, np.ndarray):
        value_flow_dtype = dtype_util.convert_numpy_dtype_to_oneflow_dtype(value.dtype)
    else:
        value_flow_dtype = value.dtype
    assert var_blob.dtype == value_flow_dtype, "{} vs {}".format(
        var_blob.dtype, value_flow_dtype
    )
    for start, stop, slice in _ReadSlice(value):
        slice_value_blob = _GetCpu0VariableBlobFromNumpy(slice, var_blob.dtype)
        _LogicalSliceAssign(
            var_blob, slice_value_blob, start, stop,
        )


@oneflow_export("load_variables")
@session_ctx.try_init_default_session
def LoadVariables(
    value_dict: Dict[str, ValueContainer], ignore_mismatch: bool = True,
):
    """
    Load value in `value_dict` into oneflow variables.
    For example, if `value_dict` is {'x', np.ones(x_shape)},
    the value of variable "x" will all ones.
    If `ignore_mismatch` is False, an exception will be raised when
    there is a name in `value_dict` not belonging to any variable.
    """
    oneflow.sync_default_session()

    all_vars = GetAllVariables()
    for name, value in value_dict.items():
        if name in all_vars:
            var_blob = interface_op_read_and_write.GetEagerInterfaceBlob(name)
            _FeedValueToVariable(var_blob, value)
        else:
            if not ignore_mismatch:
                raise RuntimeError('"{}" is not a variable name'.format(name))
    oneflow_api.eager.Sync()


def _ForEachSlice(
    container: ValueContainer,
    f: Union[
        Callable[[EagerBlobTrait, Sequence[int], Sequence[int]], Any],
        Callable[[FileBackendVariableBlob, Sequence[int], Sequence[int]], Any],
        Callable[[np.ndarray, Sequence[int], Sequence[int]], Any],
    ],
):
    """
    Slice container into slices whose size < SLICE_BYTES. For every slice,
    yield start_nd_idx, stop_nd_idx and f(slice)
    """
    assert isinstance(
        container, (EagerBlobTrait, FileBackendVariableBlob, np.ndarray)
    ), "Unknown type: {}".format(type(container).__name__)
    assert container.shape is not None
    # For current implementation (transport data by grpc), SLICE_BYTES must be lower than 64M
    SLICE_BYTES = 32 * 1024 * 1024
    if isinstance(container, np.ndarray):
        np_dtype = container.dtype
    else:
        np_dtype = np.dtype(
            dtype_util.convert_oneflow_dtype_to_numpy_dtype(container.dtype)
        )
    SLICE_LEN = SLICE_BYTES // np_dtype.itemsize
    start_idx = 0
    size = _ElemCnt(container.shape)
    cnt = 1
    for axis in reversed(range(len(container.shape))):
        cnt *= container.shape[axis]
        if cnt > SLICE_LEN:
            break
    unit_size = _ElemCnt(container.shape[axis + 1 :])
    max_unit_num = SLICE_LEN // unit_size
    while start_idx < size:
        remainder = container.shape[axis]
        while remainder > 0:
            unit_num = max_unit_num if remainder >= max_unit_num else remainder
            length = unit_num * unit_size
            remainder -= unit_num
            stop_idx = start_idx + length
            start_nd_idx = np.unravel_index(start_idx, container.shape)
            stop_nd_idx = np.unravel_index(stop_idx - 1, container.shape)
            stop_nd_idx = tuple([x + 1 for x in stop_nd_idx])
            yield start_nd_idx, stop_nd_idx, f(container, start_nd_idx, stop_nd_idx)
            start_idx = stop_idx


def Init() -> None:
    oneflow.sync_default_session()

    sess = session_ctx.GetDefaultSession()
    for op_name, var_blob in GetAllVariables().items():
        var_conf = sess.OpAttribute4InterfaceOpName(op_name).op_conf.variable_conf
        if not (
            var_conf.HasField("initializer")
            or var_conf.HasField("initialize_with_snapshot")
        ):
            continue
        if var_conf.HasField("initialize_with_snapshot"):
            initialize_with_snapshot_conf = var_conf.initialize_with_snapshot
            if initialize_with_snapshot_conf.HasField("key"):
                snapshot_key = op_name
            else:
                snapshot_key = initialize_with_snapshot_conf.key
            var_dir = os.path.dirname(
                os.path.join(initialize_with_snapshot_conf.path, snapshot_key,)
            )
            LoadVariables({op_name: GetCheckpoint(var_dir)})
            continue
        g = initializer_util.GetInitializer(
            var_conf.initializer, var_conf.random_seed, var_blob.shape
        )

        def GenerateValueAndAssign(var_blob, start_nd_idx, stop_nd_idx):
            np_dtype = np.dtype(
                dtype_util.convert_oneflow_dtype_to_numpy_dtype(var_blob.dtype)
            )
            length = _ElemCnt(np.array(stop_nd_idx) - np.array(start_nd_idx))
            vals = (
                np.array(g(length))
                .astype(np_dtype)
                .reshape(np.array(stop_nd_idx) - np.array(start_nd_idx))
            )

            slice_value_blob = _GetCpu0VariableBlobFromNumpy(vals, var_blob.dtype)
            _LogicalSliceAssign(
                var_blob, slice_value_blob, start_nd_idx, stop_nd_idx,
            )

        # we just want to run f on every slice without caring about the return value
        for _ in _ForEachSlice(var_blob, GenerateValueAndAssign):
            pass
    oneflow_api.eager.Sync()
