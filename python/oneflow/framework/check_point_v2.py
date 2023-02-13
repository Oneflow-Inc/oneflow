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
import contextlib
import os
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import pickle
import json
from collections import OrderedDict

import numpy as np
from google.protobuf import text_format

import oneflow
import oneflow as flow
import oneflow._oneflow_internal
import oneflow.core.framework.variable_meta_info_pb2 as variable_meta_info_pb
import oneflow.framework.dtype as dtype_util
import oneflow.framework.id_util as id_util
from oneflow.framework.tensor import Tensor
import oneflow.nn.graph.graph as graph_util
from oneflow.framework.args_tree import ArgsTree
import pickle
from oneflow.nn.graph import GraphTensor

SNAPSHOT_DONE_FILENAME = "snapshot_done"
META_INFO_FILENAME = "meta"
PICKLE_FILENAME = "pickled_data"
DATA_FILENAME = "out"
PROTOCOL_VERSION = 1
ONEFLOW_MAGIC_KEY = "__oneflow__"


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


def _save_tensor_to_disk(tensor: "oneflow.Tensor", dir_name: Union[str, Path]) -> None:
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


def smart_to(
    obj: Any, dest: Optional[Union[str, flow.device, flow.placement]]
) -> "oneflow.Tensor":
    if not isinstance(obj, flow.Tensor):
        return obj
    tensor = obj
    if dest is None:
        return tensor
    if isinstance(dest, (str, flow.device)):
        return tensor.to(device=dest)
    else:
        return tensor.to_global(placement=dest)


def _LoadSingleVariable(
    path: Optional[str],
    global_src_rank: Optional[int] = None,
    map_location: Optional[Union[flow.device, flow.placement]] = None,
) -> "flow.Tensor":
    if global_src_rank is not None:
        rank = flow.env.get_rank()
        if rank == global_src_rank:
            file_backed_blob = FileBackendVariableBlob(path)
            loaded = flow.tensor(file_backed_blob.numpy(), dtype=file_backed_blob.dtype)
        else:
            loaded = flow.tensor([])
        loaded = loaded.to_global(
            flow.placement("cpu", [global_src_rank]), flow.sbp.broadcast
        )
    else:
        loaded = flow.tensor(FileBackendVariableBlob(path).numpy())
    return smart_to(loaded, map_location)


def _broadcast_py_object(obj, src: int = 0):
    rank = flow.env.get_rank()
    if src == rank:
        obj_bytes = pickle.dumps(obj)
        return pickle.loads(flow._oneflow_internal.cpu_broadcast(obj_bytes, src))
    else:
        return pickle.loads(flow._oneflow_internal.cpu_broadcast(None, src))


# NOTE(jianhao):
# (de)serializing a container of global tensors requires the order
# of those tensors are the same across all ranks.
def tensor_getstate(self):
    # context_data is not None means setstate/getstate is called inside
    # flow.save or flow.load
    if context_data is not None:
        if context_data.global_rank is None:
            assert (
                self.is_local
            ), "Please set global_dst_rank in `flow.save` to save global tensor"
            tensor = self
        else:
            assert not self.is_local
            # Boxing to cpu firstly to avoid extra gpu memory usage
            tensor = (
                self.to_global(
                    sbp=self.sbp, placement=flow.placement("cpu", self.placement.ranks)
                )
                .to_global(
                    sbp=flow.sbp.broadcast,
                    placement=flow.placement("cpu", [context_data.global_rank]),
                )
                .to_local()
            )
        if context_data.save_as_external_data:
            if context_data.global_rank is None:
                rel_dir_name = id_util.UniqueStr("tensor_")
            else:
                rel_dir_name = f"global_tensor_{self.global_id()}"
            abs_dir_name = context_data.path / rel_dir_name

            if (
                context_data.global_rank is None
                or context_data.global_rank == flow.env.get_rank()
            ):
                _save_tensor_to_disk(tensor, abs_dir_name)

            return {"path": rel_dir_name}
        else:
            return {
                "data": tensor.numpy(),
                "dtype": tensor.dtype,
                "device": "cpu",
            }
    else:
        if self.is_local:
            if self.is_cuda:
                device = "cuda"
            else:
                device = "cpu"
            return {"data": self.numpy(), "dtype": self.dtype, "device": device}
        else:
            return {
                "data": self.numpy(),
                "dtype": self.dtype,
                "placement": self.placement,
                "sbp": self.sbp,
            }


def tensor_setstate(self, pickle_dict):
    if context_data is not None:
        if context_data.save_as_external_data:
            rel_dir_name = pickle_dict["path"]
            abs_dir_name = context_data.path / rel_dir_name
            tmp_tensor = _LoadSingleVariable(
                str(abs_dir_name), context_data.global_rank, context_data.map_location
            )
            self.__init__(tmp_tensor)
        else:
            self.__init__(flow.tensor(pickle_dict["data"], dtype=pickle_dict["dtype"]))
    else:
        if "placement" in pickle_dict:
            return self.__init__(
                flow.tensor(
                    pickle_dict["data"],
                    dtype=pickle_dict["dtype"],
                    placement=pickle_dict["placement"],
                    sbp=pickle_dict["sbp"],
                )
            )
        else:
            return self.__init__(
                flow.tensor(
                    pickle_dict["data"],
                    dtype=pickle_dict["dtype"],
                    device=pickle_dict["device"],
                )
            )


def placement_getstate(self):
    return {
        "type": self.type,
        "ranks": self.ranks,
    }


def placement_setstate(self, state):
    return self.__init__(state["type"], state["ranks"])


def RegisterMethods():
    Tensor.__setstate__ = tensor_setstate
    Tensor.__getstate__ = tensor_getstate
    flow._oneflow_internal.placement.__getstate__ = placement_getstate
    flow._oneflow_internal.placement.__setstate__ = placement_setstate


load_methods = []


def load_if(condition):
    def decorator(func):
        def condition_always_returning_extra_data(*args, **kwargs):
            res = condition(*args, **kwargs)
            if isinstance(res, tuple):
                assert len(res) == 2
                assert isinstance(res[1], tuple)
                return res
            else:
                return res, ()

        load_methods.append((condition_always_returning_extra_data, func))
        return func

    return decorator


def is_dir_and_no_pickle_file(path: Path, support_pytorch_format: bool):
    if path.is_dir():
        pickle_path = path / PICKLE_FILENAME
        return not pickle_path.exists()
    return False


@load_if(is_dir_and_no_pickle_file)
def legacy_load(
    path: Path,
    global_src_rank: Optional[int],
    map_location: Optional[Union[str, flow.device]],
) -> Dict[str, "flow.Tensor"]:
    assert os.path.isdir(path), "Directory {} doesn't exist!".format(path)
    rank = flow.env.get_rank()
    var_dict = {}
    if global_src_rank is None or rank == global_src_rank:
        all_files = os.listdir(path)
        assert SNAPSHOT_DONE_FILENAME in all_files
        all_files.remove(SNAPSHOT_DONE_FILENAME)
        if global_src_rank is not None:
            _broadcast_py_object(all_files, global_src_rank)
    else:
        all_files = _broadcast_py_object(None, global_src_rank)
    for f in all_files:
        var_dir = os.path.join(path, f)
        try:
            var_dict[f] = _LoadSingleVariable(var_dir, global_src_rank, map_location)
        except FileNotFoundError:
            warnings.warn(
                f"'{var_dir}' does not have valid tensor data. Please check it if it is unexpected.",
                stacklevel=2,
            )
    return var_dict


@contextlib.contextmanager
def tensor_pickling_context(
    path: Path,
    global_rank: Optional[int],
    mp: Optional[Union[str, flow.device, flow.placement]],
    save_as_external_data: bool,
):
    global context_data
    context_data = ContextData(path, global_rank, mp, save_as_external_data)
    try:
        yield
    finally:
        context_data = None


def is_oneflow_pickle_file(path: Path, support_pytorch_format: bool) -> bool:
    if not path.is_file():
        return False
    try:
        with open(path, "rb") as f:
            content = pickle.load(f)
            if ONEFLOW_MAGIC_KEY in content:
                return True, (content,)
            else:
                return False
    except:
        return False


# `path` is not used in this function, because the file is already loaded
# and deserialized in `is_oneflow_pickle_file`, and the content is passed
# as `content`.
@load_if(is_oneflow_pickle_file)
def load_from_oneflow_single_file(
    path: Path,
    global_src_rank,
    map_location: Optional[Union[str, flow.device]],
    content: Any = None,
):
    rank = flow.env.get_rank()
    if global_src_rank is None or rank == global_src_rank:
        assert content["protocol_version"] == PROTOCOL_VERSION
        res = content["data"]
    else:
        res = None

    if global_src_rank is not None:
        res = flow.utils.global_view.to_global(
            res,
            placement=flow.placement("cpu", [global_src_rank]),
            sbp=flow.sbp.broadcast,
            warn_on_non_tensor_leaf=False,
        )
    res = ArgsTree(res).map_leaf(lambda x: smart_to(x, map_location))
    return res


def is_file_and_support_pytorch_format(
    path: Path, support_pytorch_format: bool
) -> bool:
    return path.is_file() and support_pytorch_format


@load_if(is_file_and_support_pytorch_format)
def load_from_pytorch_file(
    path: Path, global_src_rank, map_location: Optional[Union[str, flow.device]],
):
    with flow.mock_torch.disable():
        import torch

        if global_src_rank is None or global_src_rank == flow.env.get_rank():
            torch_obj = torch.load(path, map_location="cpu")

            def torch_tensor_to_flow(x):
                if isinstance(x, torch.Tensor):
                    return flow.utils.tensor.from_torch(x)
                else:
                    return x

            flow_obj = ArgsTree(torch_obj).map_leaf(torch_tensor_to_flow)
        else:
            flow_obj = None
        if global_src_rank is not None:
            flow_obj = flow.utils.global_view.to_global(
                flow_obj,
                placement=flow.placement("cpu", [global_src_rank]),
                sbp=flow.sbp.broadcast,
                warn_on_non_tensor_leaf=False,
            )

        flow_obj = ArgsTree(flow_obj).map_leaf(lambda x: smart_to(x, map_location))
        return flow_obj


def is_dir_and_has_pickle_file(path: Path, support_pytorch_format: bool) -> bool:
    if path.is_dir():
        pickle_path = path / PICKLE_FILENAME
        return pickle_path.exists()
    return False


@load_if(is_dir_and_has_pickle_file)
def load_from_oneflow_pickle_dir(
    path: Path,
    global_src_rank: Optional[int],
    map_location: Optional[Union[str, flow.device, flow.placement]],
):
    rank = flow.env.get_rank()
    pickle_path = path / PICKLE_FILENAME
    if global_src_rank is not None:
        if rank == global_src_rank:
            pickle_bytes = pickle_path.read_bytes()
            _broadcast_py_object(pickle_bytes, global_src_rank)
        else:
            pickle_bytes = _broadcast_py_object(None, global_src_rank)
    else:
        pickle_bytes = pickle_path.read_bytes()

    if map_location is not None:
        assert isinstance(
            map_location, (str, flow.device, flow.placement)
        ), "'map_location' only supports str, device or placement."
    with tensor_pickling_context(path, global_src_rank, map_location, True):
        res = pickle.loads(pickle_bytes)
    assert res["protocol_version"] == PROTOCOL_VERSION
    return res["data"]


def load(
    path: str,
    global_src_rank: Optional[int] = None,
    map_location: Optional[Union[str, flow.device, flow.placement]] = None,
    *,
    support_pytorch_format: bool = True,
) -> Any:
    r"""Loads an object saved with oneflow.save() from a directory.

    Args:
        path (str): The directory containing the object
        global_src_rank (int, optional): The source rank for
            loading global tensors. When specified, only the
            process whose rank == global_src_rank will really
            read the files in `path`, and tensors in the loaded
            object will be consistent with placement =
            `flow.placement('cuda', [global_src_rank])`
        map_location (str, flow.device or flow.placement, optional):
            indicates the location where all tensors should be loaded.
        support_pytorch_format (bool, optional): whether to support
            loading the file saved by `torch.save`. Default: True

    Returns:
        The loaded object
    """
    path: Path = Path(path)
    rank = flow.env.get_rank()
    if global_src_rank is None or global_src_rank == rank:
        for i, (condition, load) in enumerate(load_methods):
            is_ok, extra_data = condition(path, support_pytorch_format)
            if is_ok:
                if global_src_rank is not None:
                    _broadcast_py_object(i, global_src_rank)
                break
        else:
            raise NotImplementedError("No valid load method found for {}".format(path))
    else:
        i = _broadcast_py_object(None, global_src_rank)
        load = load_methods[i][1]
        extra_data = ()

    return load(path, global_src_rank, map_location, *extra_data)  # type: ignore


def save_one_embedding_info(state_dict: Any, path: Union[str, Path]) -> None:
    path: Path = Path(path)

    _embedding_info_dict = {"embedding": []}
    os.makedirs(path, exist_ok=True)

    _save_one_embedding_info_flag = False

    for module in state_dict.keys():
        if not isinstance(state_dict[module], OrderedDict):
            continue
        for module_key in state_dict[module].keys():
            _info_dict = {}
            if "OneEmbeddingKeyValueOptions" in module_key:
                if not _save_one_embedding_info_flag:
                    _save_one_embedding_info_flag = True

                module_key_prefix = module_key.rstrip("OneEmbeddingKeyValueOptions")

                _embedding_info_dict["embedding"].append(
                    {
                        "snapshot": state_dict["module"][
                            module_key_prefix + "OneEmbeddingSnapshot"
                        ],
                        "kv_options": json.loads(
                            state_dict["module"][
                                module_key_prefix + "OneEmbeddingKeyValueOptions"
                            ]
                        ),
                    }
                )

    if _save_one_embedding_info_flag:
        with open(os.path.join(path, "one_embedding_options.json"), "w") as f:
            f.write(json.dumps(_embedding_info_dict, indent=4))


def save(
    obj: Any,
    path: Union[str, Path],
    global_dst_rank: Optional[int] = None,
    save_as_external_data: bool = False,
) -> None:
    r"""Save an object to a directory.

    Args:
        obj: The object to be saved
        path (str): The directory in which the object is saved
        global_dst_rank (int, optional): The destination rank for
            saving global tensors. When specified, whole tensors
            will be saved by the process whose rank ==
            global_src_rank, while other processes will not do any
            disk I/O.
    """
    path: Path = Path(path)

    if isinstance(obj, graph_util.Graph):
        graph: graph_util.Graph = obj
        if not graph._is_compiled:
            raise RuntimeError("graph must be compiled first.")

        path.mkdir(exist_ok=True)

        serialized_job = graph._forward_job_proto.SerializeToString()
        oneflow._oneflow_internal.nn.graph.SaveJobToIR(serialized_job, str(path))

        for x in graph._state():
            _save_tensor_to_disk(
                x.to(Tensor),
                path / f"{x.to(GraphTensor).name_prefix}{x.to(GraphTensor).name}",
            )

        save_one_embedding_info(obj.state_dict(), path)

        return

    obj = {"protocol_version": PROTOCOL_VERSION, ONEFLOW_MAGIC_KEY: None, "data": obj}

    with tensor_pickling_context(path, global_dst_rank, None, save_as_external_data):
        pickled_bytes = pickle.dumps(obj)

    def write_file():
        if save_as_external_data:
            path.mkdir(exist_ok=True)
            pickle_path = path / PICKLE_FILENAME
            pickle_path.write_bytes(pickled_bytes)
        else:
            path.write_bytes(pickled_bytes)

    if global_dst_rank is not None:
        assert isinstance(
            global_dst_rank, int
        ), f"global_dst_rank expected type int, but got {type(global_dst_rank)}."
        assert (
            global_dst_rank >= 0 and global_dst_rank < flow.env.get_world_size()
        ), f"out of range (expected to be in range of [0, {flow.env.get_world_size()}), but got {global_dst_rank})."
        if flow.env.get_rank() == global_dst_rank:
            write_file()
    else:
        # global_dst_rank is None
        write_file()


class ContextData:
    def __init__(
        self,
        path: Path,
        global_rank: Optional[int],
        map_location: Optional[Union[str, flow.device, flow.placement]],
        save_as_external_data: bool,
    ):
        self.path = path
        self.global_rank = global_rank
        self.map_location = map_location
        self.save_as_external_data = save_as_external_data


context_data = None
