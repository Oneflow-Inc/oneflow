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
import oneflow
import oneflow._oneflow_internal


def create_generator(device=None):
    if device is None:
        device = "auto"
    return oneflow._oneflow_internal.create_generator(device)


def seed() -> int:
    r"""
    Sets the seed for generating random numbers to a non-deterministic
    random number. Returns a 64 bit number used to seed the RNG.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.seed.html.
    """
    seed = default_generator.seed()
    oneflow._oneflow_internal.manual_seed(seed)
    return seed


def manual_seed(seed):
    r"""
    Sets the seed for generating random numbers. Returns a
    `oneflow.Generator` object.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.manual_seed.html.

    Args:
        seed (int): The desired seed. Value must be within the inclusive range
            `[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]`. Otherwise, a RuntimeError
            is raised. Negative inputs are remapped to positive values with the formula
            `0xffff_ffff_ffff_ffff + seed`.
    """
    seed = int(seed)
    return oneflow._oneflow_internal.manual_seed(seed)


def initial_seed() -> int:
    r"""
    Returns the initial seed for generating random numbers as a
    Python `long`.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/_modules/torch/random.html.
    
    """
    return default_generator.initial_seed()


def _getstate(self):
    return {"device": str(self.device), "state": self.get_state()}


def _setstate(self, state_dict):
    self.__init__(state_dict["device"])
    self.set_state(state_dict["state"])


def get_rng_state():
    r"""
    Sets the random number generator state.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.get_rng_state.html.
    
    .. note: This function only works for CPU. For CUDA, please use
             oneflow.manual_seed(seed), which works for both CPU and CUDA.

    Args:
        new_state (oneflow.ByteTensor): The desired state
    """
    return oneflow.default_generator.get_state()


def set_rng_state(state):
    """
    Returns the random number generator state as a `oneflow.ByteTensor`.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.set_rng_state.html.
    
    """

    return oneflow.default_generator.set_state(state)


default_generator = oneflow._oneflow_internal.default_generator("cpu")
oneflow._oneflow_internal.Generator.__getstate__ = _getstate
oneflow._oneflow_internal.Generator.__setstate__ = _setstate
