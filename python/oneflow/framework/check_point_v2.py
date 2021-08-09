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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from google.protobuf import text_format

import oneflow
import oneflow._oneflow_internal
import oneflow._oneflow_internal.oneflow.core.register.logical_blob_id as lbi_util
import oneflow.core.framework.user_op_attr_pb2 as attr_value_pb
import oneflow.core.framework.variable_meta_info_pb2 as variable_meta_info_pb
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.eager.boxing_util as boxing_util
import oneflow.eager.op_infer_util as op_infer_util
import oneflow.framework.config_util as config_util
import oneflow.framework.dtype as dtype_util
import oneflow.framework.id_util as id_util
import oneflow.framework.remote_blob as remote_blob_util
import oneflow.framework.runtime_mode as rt_mode
import oneflow.framework.session_context as session_ctx
import oneflow.ops.initializer_util as initializer_util
import oneflow.support.async_util as async_util
from oneflow._oneflow_internal import EagerBlobTrait
from oneflow.experimental import interface_op_read_and_write

META_INFO_FILENAME = "meta"
DATA_FILENAME = "out"
FAKE_JOB_NAME = "system_checkpoint"
OP_PREFIX = "system_checkpoint"
blob_register = oneflow._oneflow_internal.GetDefaultBlobRegister()


def sync_default_session_if_normal():
    if rt_mode.CurrentMode() == rt_mode.NORMAL_MODE:
        oneflow.sync_default_session()
    else:
        pass


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
        elif shape is not None and dtype is not None:
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


ValueContainer = Union[
    EagerBlobTrait, FileBackendVariableBlob, np.ndarray, "oneflow.Tensor"
]


def _ElemCnt(shape):
    return np.prod(shape).astype(int).item()


def _LoadSingleVariable(path: str) -> Optional[FileBackendVariableBlob]:
    if os.path.isfile(os.path.join(path, DATA_FILENAME)):
        return FileBackendVariableBlob(path)
    return None


def _GetCheckpoint(
    path: str,
) -> Union[Dict[str, FileBackendVariableBlob], FileBackendVariableBlob]:
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


@session_ctx.try_init_default_session
def GetCheckpoint(
    path: str,
) -> Union[Dict[str, FileBackendVariableBlob], FileBackendVariableBlob]:
    """
    Load variable(s) from file system.
    """
    return _GetCheckpoint(path)


def Load(
    path: str,
) -> Union[Dict[str, FileBackendVariableBlob], FileBackendVariableBlob]:
    return _GetCheckpoint(path)


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
    if isinstance(container, oneflow.Tensor):

        def ReadFromTensor(tensor, start_nd_idx, stop_nd_idx):
            start_nd_idx = list(map(int, start_nd_idx))
            stop_nd_idx = list(map(int, stop_nd_idx))
            return tensor[
                tuple(
                    [
                        slice(start_nd_idx[i], stop_nd_idx[i])
                        for i in range(len(start_nd_idx))
                    ]
                )
            ].numpy()

        yield from _ForEachSlice(container, ReadFromTensor)
    elif isinstance(container, EagerBlobTrait):

        def ReadFromEagerBlob(eager_blob, start_nd_idx, stop_nd_idx):
            scope_symbol_id = _GetScopeSymbolIdFromEagerBlob(eager_blob)
            return _LogicalSlice(
                eager_blob.blob_object, start_nd_idx, stop_nd_idx, scope_symbol_id
            )

        yield from _ForEachSlice(container, ReadFromEagerBlob)
    elif isinstance(container, FileBackendVariableBlob):
        np_dtype = np.dtype(
            dtype_util.convert_oneflow_dtype_to_numpy_dtype(container.dtype)
        )
        with open(container.file_path, "rb") as f:

            def ReadFromFile(_, start_nd_idx, stop_nd_idx):
                length = _ElemCnt(np.array(stop_nd_idx) - np.array(start_nd_idx))
                slice = f.read(length * np_dtype.itemsize)
                return np.frombuffer(slice, dtype=np_dtype).reshape(
                    np.array(stop_nd_idx) - np.array(start_nd_idx)
                )

            yield from _ForEachSlice(container, ReadFromFile)
    elif isinstance(container, np.ndarray):

        def ReadFromNpArray(array, start_nd_idx, stop_nd_idx):
            slice_objs = []
            for (start, stop) in zip(start_nd_idx, stop_nd_idx):
                slice_objs.append(slice(start, stop))
            return array[tuple(slice_objs)]

        yield from _ForEachSlice(container, ReadFromNpArray)
    else:
        raise RuntimeError("Unknown type: {}".format(type(container).__name__))


def _SaveVarDict(
    path: str,
    var_dict: Optional[
        Dict[str, Union[FileBackendVariableBlob, EagerBlobTrait]]
    ] = None,
) -> None:
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
    ), "{} is a file or non-empty directory! Note that flow.save is different from torch.save. It saves each weight as a separated file so that a directory instead of a file should be given.".format(
        path
    )
    os.makedirs(path, exist_ok=True)
    for (name, var) in var_dict.items():
        meta_info = variable_meta_info_pb.VariableMetaInfo()
        meta_info.shape.dim[:] = var.shape
        meta_info.data_type = oneflow._oneflow_internal.deprecated.GetProtoDtype4OfDtype(
            var.dtype
        )
        var_dir = os.path.join(path, name)
        param_path = os.path.join(var_dir, DATA_FILENAME)
        os.makedirs(os.path.dirname(param_path))
        with open(param_path, "wb") as f:
            for (_, _, slice) in _ReadSlice(var):
                f.write(slice.tobytes())
        with open(os.path.join(var_dir, META_INFO_FILENAME), "w") as f:
            f.write(text_format.MessageToString(meta_info))
    with open(os.path.join(path, "snapshot_done"), "w"):
        pass


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
    sync_default_session_if_normal()
    return _SaveVarDict(path, var_dict)


def save(obj, save_dir):
    return _SaveVarDict(save_dir, obj)


def _LogicalSlice(
    input_blob_object: oneflow._oneflow_internal.BlobObject,
    start: Sequence[int],
    stop: Sequence[int],
    scope_symbol_id: int,
) -> np.ndarray:
    """
    Construct a logical_slice op and run it by oneflow eager,
    return the sliced result as a numpy ndarray
    """
    op_name = id_util.UniqueStr(OP_PREFIX)

    def AsyncSlice(Yield):
        def build(builder):
            op_conf = op_conf_pb.OperatorConf()
            device_tag = oneflow.current_scope().device_parallel_desc_symbol.device_tag
            op_conf.device_tag = device_tag
            op_conf.name = op_name
            op_conf.user_conf.op_type_name = "logical_slice"
            op_conf.user_conf.input["x"].s.append("{}/x_0".format(op_name))
            op_conf.user_conf.output["y"].s.append("{}/y_0".format(op_name))
            parallel_conf = input_blob_object.parallel_desc_symbol.parallel_conf
            op_conf.user_conf.attr["parallel_conf"].at_string = str(parallel_conf)
            op_conf.user_conf.attr["start"].at_list_int64.val[:] = start
            op_conf.user_conf.attr["stop"].at_list_int64.val[:] = stop
            op_conf.user_conf.attr["step"].at_list_int64.val[:] = [1] * len(start)
            bn_in_op2blob_object = (
                oneflow._oneflow_internal.deprecated.BnInOp2BlobObject()
            )
            bn_in_op2blob_object["x_0"] = input_blob_object
            op_attribute = op_infer_util.Infer(
                op_conf, bn_in_op2blob_object, scope_symbol_id
            )
            cfg_op_attribute = oneflow._oneflow_internal.deprecated.MakeOpAttributeByString(
                str(op_attribute)
            )
            builder.StatelessCall(
                cfg_op_attribute,
                parallel_conf,
                bn_in_op2blob_object,
                boxing_util.BoxingTo,
            )
            Yield(bn_in_op2blob_object["y_0"])

        oneflow._oneflow_internal.deprecated.LogicalRun(build)

    lbi = lbi_util.LogicalBlobId()
    lbi.set_op_name(op_name)
    lbi.set_blob_name(op_name)
    blob_object = async_util.Await(1, AsyncSlice)[0]
    blob = oneflow._oneflow_internal.EagerConsistentBlob(
        lbi,
        blob_object=blob_object,
        blob_register=blob_register,
        job_name=FAKE_JOB_NAME,
    )
    return blob.numpy()


def _GetCpu0VariableBlobFromNumpy(
    np_array: np.ndarray, dtype: oneflow.dtype
) -> oneflow._oneflow_internal.EagerConsistentBlob:
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
    ref_blob_object: oneflow._oneflow_internal.BlobObject,
    value_blob_object: oneflow._oneflow_internal.BlobObject,
    start: Sequence[int],
    stop: Sequence[int],
    scope_symbol_id: Optional[int],
) -> None:
    """
    Construct a logical_slice_assign op and run it by oneflow eager
    """

    def BuildAssignInstruction(builder):
        op_conf = op_conf_pb.OperatorConf()
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
        bn_in_op2blob_object = oneflow._oneflow_internal.deprecated.BnInOp2BlobObject()
        bn_in_op2blob_object["ref_0"] = ref_blob_object
        bn_in_op2blob_object["value_0"] = value_blob_object
        op_attribute = op_infer_util.Infer(
            op_conf, bn_in_op2blob_object, scope_symbol_id
        )
        cfg_op_attribute = oneflow._oneflow_internal.deprecated.MakeOpAttributeByString(
            str(op_attribute)
        )
        builder.StatelessCall(
            cfg_op_attribute, parallel_conf, bn_in_op2blob_object, boxing_util.BoxingTo
        )

    oneflow._oneflow_internal.deprecated.LogicalRun(BuildAssignInstruction)


def FeedValueToVariable(
    var_blob: Union[oneflow._oneflow_internal.EagerConsistentBlob, "oneflow.Tensor"],
    value: ValueContainer,
    scope_symbol_id: Optional[int],
) -> None:
    """
    Feed the value of `value` to the variable `var_blob`
    """
    assert isinstance(
        value, (EagerBlobTrait, FileBackendVariableBlob, np.ndarray, oneflow.Tensor)
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
    if isinstance(var_blob, oneflow.Tensor):
        raise ValueError("Tensor object arguments are not supported")
    else:
        assert isinstance(var_blob, EagerBlobTrait)
        var_blob_object = var_blob.blob_object
    for (start, stop, slice) in _ReadSlice(value):
        slice_value_blob = _GetCpu0VariableBlobFromNumpy(slice, var_blob.dtype)
        _LogicalSliceAssign(
            var_blob_object, slice_value_blob.blob_object, start, stop, scope_symbol_id
        )


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
        container, (EagerBlobTrait, FileBackendVariableBlob, np.ndarray, oneflow.Tensor)
    ), "Unknown type: {}".format(type(container).__name__)
    assert container.shape is not None
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
    unit_size = _ElemCnt(tuple(container.shape)[axis + 1 :])
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
            yield (start_nd_idx, stop_nd_idx, f(container, start_nd_idx, stop_nd_idx))
            start_idx = stop_idx


def generate_values_by_initializer(initializer, shape, dtype):
    np_dtype = np.dtype(dtype_util.convert_oneflow_dtype_to_numpy_dtype(dtype))
    length = _ElemCnt(shape)
    return np.array(initializer(length)).astype(np_dtype).reshape(shape)


def init_by_initializer_conf(
    var_blob: Union[EagerBlobTrait, "oneflow.Tensor"],
    initializer_conf: initializer_conf_util.InitializerConf,
    sync_between_multi_machine: bool,
    scope_symbol_id: Optional[int],
    random_seed: int = 0,
):
    initializer = initializer_util.GetInitializer(
        initializer_conf, random_seed, var_blob.shape
    )
    if initializer is None:
        return

    def GenerateValueAndAssign(var_blob, start_nd_idx, stop_nd_idx):
        shape = np.array(stop_nd_idx) - np.array(start_nd_idx)
        vals = generate_values_by_initializer(initializer, shape, var_blob.dtype)
        if isinstance(var_blob, oneflow.Tensor):
            raise ValueError("Tensor object arguments are not supported")
        else:
            assert isinstance(var_blob, EagerBlobTrait)
            var_blob_object = var_blob.blob_object
        slice_value_blob = _GetCpu0VariableBlobFromNumpy(vals, var_blob.dtype)
        _LogicalSliceAssign(
            var_blob_object,
            slice_value_blob.blob_object,
            start_nd_idx,
            stop_nd_idx,
            scope_symbol_id,
        )

    for _ in _ForEachSlice(var_blob, GenerateValueAndAssign):
        pass
    if sync_between_multi_machine:
        oneflow._oneflow_internal.eager.single_client.Sync()


def Init() -> None:
    raise NotImplemented()
