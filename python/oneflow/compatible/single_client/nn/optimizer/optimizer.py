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
import collections
import warnings
from typing import Any, Callable, Dict, Iterator, Union

from oneflow.compatible.single_client.framework.tensor import Tensor
from oneflow.compatible.single_client.nn.parameter import Parameter


class ParamGroup(object):
    def __init__(
        self,
        parameters: Union[Iterator[Parameter], Dict[str, Any]],
        default_options: Dict,
    ):
        if isinstance(parameters, collections.abc.Iterator):
            self._parameters = list(parameters)
            self._options = default_options
        else:
            assert "params" in parameters
            self._parameters = list(parameters["params"])
            self._options = default_options
            for key in self._options:
                if key in parameters:
                    self._options[key] = parameters[key]

    def __getitem__(self, key):
        return self._options[key]

    def __setitem__(self, key, value):
        self._options[key] = value

    @property
    def options(self):
        return self._options

    @property
    def parameters(self):
        return self._parameters


class Optimizer(object):
    def __init__(self):
        self.param_groups = list()
        self._default_options = dict()
        self._state = dict()
        self._state["step"] = 0
        self._op = None

    def add_param_group(self, param_group) -> None:
        raise NotImplementedError()

    def load_state_dict(self, state_dict) -> None:
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()

    def step(self, closure: Union[Callable, None] = None) -> Union[Tensor, None]:
        raise NotImplementedError()

    def zero_grad(self, set_to_none: bool = False):
        """Sets the gradients of all optimized torch.Tensor s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly
                improve performance. However, it changes certain behaviors.
        For example:
            1. When the user tries to access a gradient and perform manual ops on
            it, a None attribute or a Tensor full of 0s will behave differently.

            2. If the user requests zero_grad(set_to_none=True) followed by a
            backward pass, grads are guaranteed to be None for params that did not
            receive a gradient.

            3. Optimizers have a different behavior if the gradient is 0 or None
            (in one case it does the step with a gradient of 0 and in the other
            it skips the step altogether).

        Returns:
            None

        """
        all_grad_is_none = True
        for param_group in self.param_groups:
            for param in param_group.parameters:
                if param.grad is not None:
                    all_grad_is_none = False
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.zeros_()
        if all_grad_is_none:
            warnings.warn(
                "\nParameters in optimizer do not have gradient.\nPlease check `loss.backward()` is called or not,\nor try to declare optimizer after calling `module.to()`"
            )
