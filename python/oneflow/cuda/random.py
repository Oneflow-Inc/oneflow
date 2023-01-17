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
from oneflow import Tensor
from typing import Union, List
from . import current_device, device_count


def get_rng_state(device: Union[int, str, flow.device] = "cuda") -> Tensor:
    r"""Returns the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (flow.device or int, optional): The device to return the RNG state of.
            Default: ``'cuda'`` (i.e., ``flow.device('cuda')``, the current CUDA device).
    """
    # TODO (add lazy initialization mechanism in OneFlow)
    # _lazy_init()
    if isinstance(device, str):
        device = flow.device(device)
    elif isinstance(device, int):
        device = flow.device("cuda", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = flow.cuda.default_generators[idx]
    return default_generator.get_state()


def get_rng_state_all() -> List[Tensor]:
    r"""Returns a list of ByteTensor representing the random number states of all devices."""

    results = []
    for i in range(device_count()):
        results.append(get_rng_state(i))
    return results
