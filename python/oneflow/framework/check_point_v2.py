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
import oneflow.core.framework.variable_meta_info_pb2 as variable_meta_info_pb
import oneflow.framework.dtype as dtype_util

META_INFO_FILENAME = "meta"
DATA_FILENAME = "out"
FAKE_JOB_NAME = "system_checkpoint"
OP_PREFIX = "system_checkpoint"


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


ValueContainer = Union[FileBackendVariableBlob, np.ndarray, "oneflow.Tensor"]


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
    path: str, var_dict: Optional[Dict[str, FileBackendVariableBlob]] = None,
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


def SaveVarDict(
    path: str, var_dict: Optional[Dict[str, FileBackendVariableBlob]] = None,
) -> None:
    """
    Save `var_dict` to `path`
    """
    return _SaveVarDict(path, var_dict)


def save(obj, save_dir):
    return _SaveVarDict(save_dir, obj)


def _ForEachSlice(
    container: ValueContainer,
    f: Union[
        Callable[[Sequence[int], Sequence[int]], Any],
        Callable[[FileBackendVariableBlob, Sequence[int], Sequence[int]], Any],
        Callable[[np.ndarray, Sequence[int], Sequence[int]], Any],
    ],
):
    """
    Slice container into slices whose size < SLICE_BYTES. For every slice,
    yield start_nd_idx, stop_nd_idx and f(slice)
    """
    assert isinstance(
        container, (FileBackendVariableBlob, np.ndarray, oneflow.Tensor)
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
