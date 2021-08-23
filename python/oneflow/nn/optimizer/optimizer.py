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
from copy import deepcopy
from typing import Any, Callable, Dict, Union

from oneflow.framework.tensor import Tensor
from oneflow.nn.parameter import Parameter
from oneflow.nn.utils.clip_grad import clip_grad_norm_


class ParamGroup(object):
    def __init__(
        self, parameters: Dict[str, Any], default_options: Dict,
    ):
        # ParamGroup must be constructed by Dict["params": parameters: List[Parameter or Tensor], "...": ...]
        assert isinstance(parameters, dict) and "params" in parameters
        assert not isinstance(parameters["params"], (Parameter, Tensor))
        self._parameters = list(parameters["params"])
        self._options = deepcopy(default_options)
        for key in self._options:
            if key in parameters:
                self._options[key] = parameters[key]
        self._enable_clip_grad = False
        if "clip_grad_max_norm" in parameters and "clip_grad_norm_type" in parameters:
            self._enable_clip_grad = True
            self._options["clip_grad_max_norm"] = parameters["clip_grad_max_norm"]
            self._options["clip_grad_norm_type"] = parameters["clip_grad_norm_type"]

    def __getitem__(self, key):
        return self._options[key]

    def __setitem__(self, key, value):
        self._options[key] = value

    def __contains__(self, key):
        return self._options.__contains__(key)

    @property
    def options(self):
        return self._options

    @property
    def parameters(self):
        return self._parameters


class Optimizer(object):
    def __init__(self, parameters, options):
        self.param_groups = list()
        self._default_options = options
        self._state = dict()
        self._state["step"] = 0

        self._parse_input_parameters(parameters)

    def add_param_group(self, param_group) -> None:
        raise NotImplementedError()

    def load_state_dict(self, state_dict) -> None:
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()

    def step(self, closure: Union[Callable, None] = None) -> Union[Tensor, None]:
        raise NotImplementedError()

    def clip_grad(self):
        r"""Clips the gradient of parameters in param_groups.
        """
        for param_group in self.param_groups:
            if param_group._enable_clip_grad:
                clip_grad_norm_(
                    param_group.parameters,
                    param_group["clip_grad_max_norm"],
                    param_group["clip_grad_norm_type"],
                    True,
                )
            else:
                warnings.warn(
                    "To enable clip_grad, passing the `clip_grad_max_norm` and `clip_grad_norm_type` parameters when instantializing the Optimizer."
                )

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
                "\nParameters in optimizer do not have gradient.\nPlease check `loss.backward()` is called"
                "or not,\nor try to declare optimizer after calling `module.to()`"
            )

    def _parse_input_parameters(self, parameters):
        """
        Supports such parameters:
            1. Iterator: flow.optim.SGD(module.parameters(), lr=0.1)
            2. List[Dict]: flow.optim.SGD([{"params": module1.parameters()}, {"params": module2.parameters()}])
            3. List[Parameter or Tensor]: flow.optim.SGD([module.weight, module.bias])
        """
        if isinstance(parameters, collections.abc.Iterator):
            # Iterator
            self.param_groups.append(
                ParamGroup({"params": list(parameters)}, self._default_options)
            )
        elif isinstance(parameters, collections.abc.Iterable):
            # List[Dict]
            if isinstance(parameters[0], dict):
                for param in parameters:
                    assert isinstance(param, dict)
                    self.param_groups.append(ParamGroup(param, self._default_options))
            # List[Parameter or Tensor]
            else:
                self.param_groups.append(
                    ParamGroup({"params": parameters}, self._default_options)
                )
        else:
            raise TypeError(
                f"params argument given to the optimizer should be an iterable of Tensors or dicts, but got {type(parameters)}"
            )

    def _generate_grad_clip_conf_for_optim_conf(self, param_group, optimizer_conf):
        if param_group._enable_clip_grad:
            if (
                param_group["clip_grad_max_norm"] == 1.0
                and param_group["clip_grad_norm_type"] == 2.0
            ):
                optimizer_conf.mutable_clip_conf().mutable_clip_by_global_norm().set_clip_norm(
                    param_group["clip_grad_max_norm"]
                )
            else:
                warnings.warn(
                    "For now, nn.Graph only support clip grad with `clip_grad_max_norm == 1.0` and `clip_grad_norm_type == 2.0`."
                )
