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
from multiprocessing.reduction import ForkingPickler

import numpy as np

import oneflow as flow
from oneflow.nn.parameter import Parameter
from oneflow.framework.tensor import Tensor
from oneflow.multiprocessing import shared_memory


try:
    # Early load resource_sharer to prevent a partially initialized instance
    # from being inherited in a forked child process. The reduce_storage method
    # requires this module indirectly through DupFd(). The built-in mp.Queue
    # class pickles arguments in a background thread which may overlap with the
    # fork.
    import multiprocessing.resource_sharer
except ImportError:
    pass


def rebuild_empty_tensor(shape, dtype, requires_grad):
    t = flow.tensor([], dtype=dtype)
    t.requires_grad = requires_grad
    return t.reshape(*shape)


def rebuild_shm_tensor(shm, shape, dtype, requires_grad):
    def delete_shm():
        try:
            # For unknown reasons delete_shm called in dataloader may fail
            # with "StopIteration".
            # An example is when dataloader is wrapped in a generator like
            # `log_every`.
            shm.close()
            shm.unlink()
        except:
            pass

    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    t = flow.from_numpy(arr)
    t._register_storage_delete_hook(delete_shm)
    t.requires_grad = requires_grad

    return t


def rebuild_empty_parameter(shape, dtype, requires_grad):
    t = flow.tensor([], dtype=dtype)
    t = t.reshape(*shape)
    return Parameter(t, requires_grad=requires_grad)


def rebuild_shm_parameter(shm, shape, dtype, requires_grad):
    def delete_shm():
        shm.close()
        shm.unlink()

    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    t = flow.from_numpy(arr)
    t._register_storage_delete_hook(delete_shm)
    return Parameter(t, requires_grad=requires_grad)


def reduce_tensor(tensor):
    tensor_data = tensor.numpy()
    requires_grad = tensor.requires_grad

    if tensor_data.nbytes == 0:
        return (rebuild_empty_tensor, (tensor.shape, tensor.dtype, requires_grad))
    else:
        shm = shared_memory.SharedMemory(create=True, size=tensor_data.nbytes)
        shm_numpy = np.ndarray(
            tensor_data.shape, dtype=tensor_data.dtype, buffer=shm.buf
        )
        shm_numpy[:] = tensor_data[:]
        return (
            rebuild_shm_tensor,
            (shm, tensor_data.shape, tensor_data.dtype, requires_grad),
        )


def reduce_parameter(tensor):
    tensor_data = tensor.numpy()
    requires_grad = tensor.requires_grad

    if tensor_data.nbytes == 0:
        return (rebuild_empty_parameter, (tensor, shape, tensor.dtype, requires_grad))
    else:
        shm = shared_memory.SharedMemory(create=True, size=tensor_data.nbytes)
        shm_numpy = np.ndarray(
            tensor_data.shape, dtype=tensor_data.dtype, buffer=shm.buf
        )
        shm_numpy[:] = tensor_data[:]
        return (
            rebuild_shm_parameter,
            (shm, tensor_data.shape, tensor_data.dtype, requires_grad),
        )


def init_reductions():
    ForkingPickler.register(Tensor, reduce_tensor)
    ForkingPickler.register(flow._oneflow_internal.Tensor, reduce_tensor)
    ForkingPickler.register(Parameter, reduce_parameter)
    ForkingPickler.register(flow._oneflow_internal.nn.Parameter, reduce_parameter)
