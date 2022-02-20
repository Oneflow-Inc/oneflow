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
import os
import sys

import numpy as np

import oneflow as flow
from oneflow.nn.parameter import Parameter
from oneflow.framework.tensor import Tensor
from oneflow.multiprocessing import shared_memory
from oneflow.utils.data import dataloader


try:
    # Early load resource_sharer to prevent a partially initialized instance
    # from being inherited in a forked child process. The reduce_storage method
    # requires this module indirectly through DupFd(). The built-in mp.Queue
    # class pickles arguments in a background thread which may overlap with the
    # fork.
    import multiprocessing.resource_sharer
except ImportError:
    pass


def rebuild_shm_tensor(shm, shape, dtype, requires_grad):
    def delete_shm():
        shm.close()
        try:
            shm.unlink()
        except:
            pass

    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    t = flow.from_numpy(arr)
    t._register_storage_delete_hook(delete_shm)
    t.requires_grad = requires_grad

    return t


def rebuild_shm_parameter(shm, shape, dtype, requires_grad):
    def delete_shm():
        shm.close()
        try:
            shm.unlink()
        except:
            pass

    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    t = flow.from_numpy(arr)
    t._register_storage_delete_hook(delete_shm)
    return Parameter(t, requires_grad=requires_grad)


shm_name_list = []

def cleanup_shm_at_exit(num, frame):
    for shm_name in shm_name_list:
        try:
            flow._oneflow_internal.multiprocessing.SharedMemory.unlink_by_name(shm_name)
        except:
            pass
    sys.exit()


def get_reduce_fn(rebuild_fn):
    def reduce_tensor(tensor):
        tensor_data = tensor.numpy()
        
        prefetch_factor = dataloader.get_worker_info().prefetch_factor
        while True:
            try:
                shm_name = "aofshm_" + flow._oneflow_internal.str_util.GenAlphaNumericString(8)
                if len(shm_name_list) == prefetch_factor + 2:
                    shm_name_list.pop(0)
                shm_name_list.append(shm_name)
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=tensor_data.nbytes)
                break
            except FileExistsError:
                shm_name_list.pop()
        # TODO: There will be a problem if dataloader is not the only one pickling
        # tensors by ForkingPickler.
        # The next step is generate the share memory in dataloader and maintain
        # the shm_list also in dataloader, instead of here
        shm_numpy = np.ndarray(tensor_data.shape, dtype=tensor_data.dtype, buffer=shm.buf)
        shm_numpy[:] = tensor_data[:]

        requires_grad = tensor.requires_grad
        return (
            rebuild_fn,
            (shm, tensor_data.shape, tensor_data.dtype, requires_grad),
        )

    return reduce_tensor


def init_reductions():
    ForkingPickler.register(Tensor, get_reduce_fn(rebuild_shm_tensor))
    ForkingPickler.register(Parameter, get_reduce_fn(rebuild_shm_parameter))
