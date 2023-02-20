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
# Just for alignment with pytorch, not really useful
from .constants import default_pg_timeout

from typing import List, Optional

import oneflow as flow


class ReduceOp:
    """Reduce operation enum. Mainly for PyTorch compatibility.
    Currently only support SUM.

    See also :func:`oneflow.comm.all_reduce()`
    """

    SUM = "sum"


def is_initialized() -> bool:
    """Always returns True. This function is only for PyTorch compatibility.

    Returns:
        True
    """
    return True


# PyTorch doesn't have torch.distributed.get_local_rank,
# we add it for the consistency between flow.env and flow.distributed
get_local_rank = flow.env.get_local_rank


def get_rank(group=None) -> int:
    """Alias of `oneflow.env.get_rank()` for PyTorch compatibility.

    See also :func:`oneflow.env.get_rank()`
    """
    assert group is None, "group is not supported yet"
    return flow.env.get_rank()


def get_world_size(group=None) -> int:
    """Alias of `oneflow.env.get_world_size()` for PyTorch compatibility.

    See also :func:`oneflow.env.get_world_size()`
    """
    assert group is None, "group is not supported yet"
    return flow.env.get_world_size()


def send(tensor: flow.Tensor, dst: int, group=None, tag: int = 0) -> None:
    """Alias of `oneflow.comm.send()` for PyTorch compatibility.

    See also :func:`oneflow.comm.send()`
    """
    assert group is None, "group is not supported yet"
    assert tag == 0, "tag is not supported yet"
    return flow.comm.send(tensor, dst)


def recv(tensor: flow.Tensor, src: int, group=None, tag: int = 0) -> None:
    """Alias of `oneflow.comm.recv()` for PyTorch compatibility.

    See also :func:`oneflow.comm.recv()`
    """
    assert group is None, "group is not supported yet"
    assert tag == 0, "tag is not supported yet"
    return flow.comm.recv(tensor, src)


def broadcast(
    tensor: flow.Tensor, src: int, group=None, async_op: bool = False
) -> None:
    """Alias of `oneflow.comm.broadcast()` for PyTorch compatibility.

    See also :func:`oneflow.comm.broadcast()`
    """
    assert group is None, "group is not supported yet"
    assert async_op is False, "async_op is not supported yet"
    return flow.comm.broadcast(tensor, src)


def barrier(group=None, async_op=False, device_ids=None) -> None:
    """Alias of `oneflow.comm.barrier()` for PyTorch compatibility.

    See also :func:`oneflow.comm.barrier()`
    """
    assert group is None, "group is not supported yet"
    assert async_op is False, "async_op is not supported yet"
    assert device_ids is None, "device_ids is not supported yet"
    return flow.comm.barrier()


def all_reduce(
    tensor: flow.Tensor, op: ReduceOp, group=None, async_op: bool = False
) -> None:
    """Alias of `oneflow.comm.all_reduce()` for PyTorch compatibility.

    See also :func:`oneflow.comm.all_reduce()`
    """
    assert op == ReduceOp.SUM, "only ReduceOp.SUM is supported"
    assert group is None, "group is not supported yet"
    assert async_op is False, "async_op is not supported yet"
    return flow.comm.all_reduce(tensor)


def all_gather(
    tensor_list: List[flow.Tensor],
    tensor: flow.Tensor,
    group=None,
    async_op: bool = False,
) -> None:
    """Alias of `oneflow.comm.all_gather()` for PyTorch compatibility.

    See also :func:`oneflow.comm.all_gather()`
    """
    assert group is None, "group is not supported yet"
    assert async_op is False, "async_op is not supported yet"
    return flow.comm.all_gather(tensor_list, tensor)


def reduce(
    tensor: flow.Tensor, dst: int, op: ReduceOp, group=None, async_op: bool = False
) -> None:
    """Alias of `oneflow.comm.reduce()` for PyTorch compatibility.

    See also :func:`oneflow.comm.reduce()`
    """
    assert op == ReduceOp.SUM, "only ReduceOp.SUM is supported"
    assert group is None, "group is not supported yet"
    assert async_op is False, "async_op is not supported yet"
    return flow.comm.reduce(tensor, dst)


def all_to_all(
    output_tensor_list: List[flow.Tensor],
    input_tensor_list: List[flow.Tensor],
    group=None,
    async_op: bool = False,
) -> None:
    """Alias of `oneflow.comm.all_to_all()` for PyTorch compatibility.

    See also :func:`oneflow.comm.all_to_all()`
    """
    assert group is None, "group is not supported yet"
    assert async_op is False, "async_op is not supported yet"
    return flow.comm.all_to_all(output_tensor_list, input_tensor_list)


def reduce_scatter(
    output: flow.Tensor,
    input_list: List[flow.Tensor],
    op: ReduceOp,
    group=None,
    async_op: bool = False,
) -> None:
    """Alias of `oneflow.comm.reduce_scatter()` for PyTorch compatibility.

    See also :func:`oneflow.comm.reduce_scatter()`
    """
    assert op == ReduceOp.SUM, "only ReduceOp.SUM is supported"
    assert group is None, "group is not supported yet"
    assert async_op is False, "async_op is not supported yet"
    return flow.comm.reduce_scatter(output, input_list)


def gather(
    tensor: flow.Tensor,
    gather_list: Optional[List[flow.Tensor]] = None,
    dst: int = 0,
    group=None,
    async_op: bool = False,
) -> None:
    """Alias of `oneflow.comm.gather()` for PyTorch compatibility.

    See also :func:`oneflow.comm.gather()`
    """
    assert group is None, "group is not supported yet"
    assert async_op is False, "async_op is not supported yet"
    return flow.comm.gather(tensor, gather_list, dst)


def is_available():
    return True
