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
import oneflow as flow

from oneflow.cuda.type_tensor import *


def is_available() -> bool:
    r"""Returns a bool indicating if CUDA is currently available."""
    # This function never throws and returns 0 if driver is missing or can't
    # be initialized
    return device_count() > 0


def device_count() -> int:
    r"""Returns the number of GPUs available."""
    return flow._oneflow_internal.CudaGetDeviceCount()


def current_device() -> int:
    r"""Returns local rank as device index."""
    return flow._oneflow_internal.GetCudaDeviceIndex()


def manual_seed_all(seed) -> None:
    r"""The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.cuda.manual_seed_all.html.
    
    Sets the seed for generating random numbers on all GPUs.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)
    flow._oneflow_internal.ManualSeedAllCudaGenerator(seed)


def manual_seed(seed: int) -> None:
    r"""The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.cuda.manual_seed.html.
    
    Sets the seed for generating random numbers for the current GPU.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)
    idx = current_device()
    flow._oneflow_internal.manual_seed(seed, "cuda", idx)
