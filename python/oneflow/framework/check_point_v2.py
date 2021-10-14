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
from contextlib import contextmanager
import os
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import pickle

import numpy as np
from google.protobuf import text_format

import oneflow
import oneflow as flow
import oneflow._oneflow_internal
import oneflow.core.framework.variable_meta_info_pb2 as variable_meta_info_pb
import oneflow.framework.dtype as dtype_util
import oneflow.framework.id_util as id_util
import pickle

SNAPSHOT_DONE_FILENAME = "snapshot_done"
META_INFO_FILENAME = "meta"
PICKLE_FILENAME = "pickled_data"
DATA_FILENAME = "out"
PROTOCOL_VERSION = 1


class FileBackendVariableBlob:
    def __init__(
        self,
        var_dir: str,
        dtype: Optional[oneflow.dtype] = None,
        shape: Optional[Sequence[int]] = None,
    ):
        data_path = os.path.join(var_dir, DATA_FILENAME)
        if not os.path.isfile(data_path):
            raise FileNotFoundError()
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


def _save_tensor_to_disk(tensor: "oneflow.Tensor", dir_name: str) -> None:
    os.makedirs(dir_name, exist_ok=True)
    meta_info = variable_meta_info_pb.VariableMetaInfo()
    meta_info.shape.dim[:] = tensor.shape
    meta_info.data_type = oneflow._oneflow_internal.deprecated.GetProtoDtype4OfDtype(
        tensor.dtype
    )
    data_path = os.path.join(dir_name, DATA_FILENAME)
    with open(data_path, "wb") as f:
        f.write(tensor.numpy().tobytes())

    with open(os.path.join(dir_name, META_INFO_FILENAME), "w") as f:
        f.write(text_format.MessageToString(meta_info))


ValueContainer = Union[FileBackendVariableBlob, np.ndarray, "oneflow.Tensor"]


def _ElemCnt(shape):
    return np.prod(shape).astype(int).item()


def _LoadSingleVariable(
    path: Optional[str], consistent_src_rank: Optional[int] = None
) -> "flow.Tensor":
    if consistent_src_rank is not None:
        rank = flow.env.get_rank()
        if rank == consistent_src_rank:
            assert isinstance(path, str)
            file_backed_blob = FileBackendVariableBlob(path)
            loaded = flow.tensor(
                file_backed_blob.numpy(), dtype=file_backed_blob.dtype
            ).to("cuda")
        else:
            loaded = flow.tensor([]).to("cuda")
        loaded = loaded.to_consistent(
            flow.placement("cuda", [consistent_src_rank]), flow.sbp.broadcast
        )
        return loaded

    assert isinstance(path, str)
    return flow.tensor(FileBackendVariableBlob(path).numpy())


def _broadcast_py_object(obj, src: int = 0):
    rank = flow.env.get_rank()
    if src == rank:
        obj_bytes = pickle.dumps(obj)
        return pickle.loads(flow._oneflow_internal.cpu_broadcast(obj_bytes, src))
    else:
        return pickle.loads(flow._oneflow_internal.cpu_broadcast(None, src))


# NOTE(jianhao): 
# 1. tensor_getstate and tensor_setstate is binded to tensor.__getstate__ 
# and __setstate__ in tensor.py, and can only be called inside flow.save
# or flow.load
# 2. (de)serializing a container of consistent tensors requires the order
# of those tensors are the same across all ranks.
def tensor_getstate(self):
    assert isinstance(current_path, Path)
    if current_consistent_src_dst_rank is None:
        assert self.is_local
        rel_dir_name = id_util.UniqueStr("tensor_")
        abs_dir_name = current_path / rel_dir_name

        tensor = self
    else:
        assert not self.is_local
        rel_dir_name = f'consistent_tensor_{self.consistent_id()}'
        abs_dir_name = current_path / rel_dir_name

        tensor = self.to_consistent(sbp=[flow.sbp.broadcast] * len(self.sbp)).to_local()
    if current_consistent_src_dst_rank is None or current_consistent_src_dst_rank == flow.env.get_rank():
        _save_tensor_to_disk(tensor, abs_dir_name)

    return {"path": rel_dir_name}


def tensor_setstate(self, pickle_dict):
    assert isinstance(current_path, Path)
    rel_dir_name = pickle_dict['path']
    abs_dir_name = current_path / rel_dir_name
    self.__init__(_LoadSingleVariable(str(abs_dir_name), current_consistent_src_dst_rank))


def legacy_load(
    path: Union[str, Path], consistent_src_rank: Optional[int] = None,
) -> Dict[str, "flow.Tensor"]:
    assert os.path.isdir(path), "Directory {} doesn't exist!".format(path)
    rank = flow.env.get_rank()
    var_dict = {}
    if consistent_src_rank is None or rank == consistent_src_rank:
        all_files = os.listdir(path)
        assert SNAPSHOT_DONE_FILENAME in all_files
        all_files.remove(SNAPSHOT_DONE_FILENAME)
        if consistent_src_rank is not None:
            _broadcast_py_object(all_files, consistent_src_rank)
    else:
        all_files = _broadcast_py_object(None, consistent_src_rank)
    for f in all_files:
        var_dir = os.path.join(path, f)
        try:
            var_dict[f] = _LoadSingleVariable(var_dir, consistent_src_rank)
        except FileNotFoundError:
            warnings.warn(
                f"'{var_dir}' does not have valid tensor data. Please check it if it is unexpected.",
                stacklevel=2,
            )
    return var_dict


@contextmanager
def tensor_pickling_context(path: Path, consistent_src_dst_rank: int):
    global current_path
    global current_consistent_src_dst_rank
    current_consistent_src_dst_rank = consistent_src_dst_rank
    current_path = path
    try:
        yield
    finally:
        current_consistent_src_dst_rank = None
        current_path = None


def load(
    path: str, consistent_src_rank: Optional[int] = None,
) -> Dict[str, "flow.Tensor"]:
    path: Path = Path(path)
    assert path.is_dir(), "Directory {} doesn't exist!".format(path)
    pickle_path = path / PICKLE_FILENAME
    rank = flow.env.get_rank()
    if consistent_src_rank is None or consistent_src_rank == rank:
        is_legacy = not pickle_path.exists()
        if consistent_src_rank is not None:
            _broadcast_py_object(is_legacy, consistent_src_rank)
    else:
        is_legacy = _broadcast_py_object(None, consistent_src_rank)
    if is_legacy:
        return legacy_load(path, consistent_src_rank)

    if consistent_src_rank is not None:
        if rank == consistent_src_rank:
            pickle_bytes = pickle_path.read_bytes()
            _broadcast_py_object(pickle_bytes, consistent_src_rank)
        else:
            pickle_bytes = _broadcast_py_object(None, consistent_src_rank)
    else:
        pickle_bytes = pickle_path.read_bytes()

    with tensor_pickling_context(path, consistent_src_rank):
        res = pickle.loads(pickle_bytes)
    assert res['protocol_version'] == PROTOCOL_VERSION
    return res['data']


def save(
    obj: Any,
    path: Union[str, Path],
    consistent_dst_rank: Optional[int] = None,
) -> None:
    path: Path = Path(path)
    obj = {'protocol_version': PROTOCOL_VERSION, 'data': obj}
    with tensor_pickling_context(path, consistent_dst_rank):
        pickled_bytes = pickle.dumps(obj)
    rank = flow.env.get_rank()
    if consistent_dst_rank is None or consistent_dst_rank == rank:
        path.mkdir(exist_ok=True)
        pickle_path = path / PICKLE_FILENAME
        pickle_path.write_bytes(pickled_bytes)


def generate_values_by_initializer(initializer, shape, dtype):
    np_dtype = np.dtype(dtype_util.convert_oneflow_dtype_to_numpy_dtype(dtype))
    length = _ElemCnt(shape)
    return np.array(initializer(length)).astype(np_dtype).reshape(shape)


current_path = None
current_consistent_src_dst_rank = None
