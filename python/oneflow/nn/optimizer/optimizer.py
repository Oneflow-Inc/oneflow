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
from itertools import chain
from typing import Any, Callable, Dict, Union

from oneflow.framework.tensor import Tensor
from oneflow.nn.graph.block import TensorBlock
from oneflow.nn.parameter import Parameter
from oneflow.nn.utils.clip_grad import clip_grad_norm_
import oneflow as flow


class ParamGroup(object):
    def __init__(
        self, parameters: Dict[str, Any], default_options: Dict,
    ):
        # ParamGroup must be constructed by Dict["params": parameters: List[Parameter, Tensor or TensorBlock], "...": ...]
        assert isinstance(parameters, dict) and "params" in parameters
        assert not isinstance(parameters["params"], (Parameter, Tensor))
        self._parameters = list()
        for p in parameters["params"]:
            if isinstance(p, (Parameter, Tensor)):
                self._parameters.append(p)
            elif isinstance(p, TensorBlock):
                # Add parameter from nn.Graph
                self._parameters.append(p.origin)
            else:
                raise ValueError(
                    "parameters in ParamGroup must be Tensor or TensorBlock."
                )

        self._options = deepcopy(default_options)
        # rewrite options in default_options
        for key in self._options:
            if key in parameters:
                self._options[key] = parameters[key]
        # add excess keys in dict
        for key in parameters:
            if key not in self._options and key != "params":
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

    def setdefault(self, key, value):
        if key not in self._options:
            self._options[key] = value

    def items(self):
        return self.__dict__.items()

    @property
    def options(self):
        return self._options

    @property
    def parameters(self):
        return self._parameters


class _SourceOpOnlyResourceDependenceMode:
    def __init__(self):
        self.guard_ = None

    def __enter__(self):
        self.guard = (
            flow._oneflow_internal.eager.SourceOpOnlyResourceDependenceModeGuard()
        )

    def __exit__(self, *args, **kwargs):
        del self.guard


def _decorate_step(step):
    def decorated_step(*args, **kwargs):
        with _SourceOpOnlyResourceDependenceMode():
            return step(*args, **kwargs)

    return decorated_step


class Optimizer(object):
    def __init__(self, parameters, options):
        self.param_groups = list()
        self._default_options = options
        self._state = dict()
        self._state["step"] = 0

        self._parse_input_parameters(parameters)

        self.step = _decorate_step(self.step)

    def add_param_group(self, param_group) -> None:
        raise NotImplementedError()

    def load_state_dict(self, state_dict) -> None:
        r"""
        Load the state of the optimizer which is created by `state_dict` function.

        It almost copied from: https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.load_state_dict
        """

        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of parameter groups"
            )
        param_lens = (len(g._parameters) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable((g["params"] for g in saved_groups)),
                chain.from_iterable((g._parameters for g in groups)),
            )
        }

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device or placement of param."""
            if isinstance(value, Tensor):
                if value.is_local:
                    value = value.to(param.device)
                else:
                    value = value.to_global(placement=param.placement, sbp=param.sbp)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, collections.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = dict()
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v
        self._state = state

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            group._options = deepcopy(new_group["_options"])
            group._enable_clip_grad = new_group["_enable_clip_grad"]
            return group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.param_groups = param_groups

    def state_dict(self):
        r"""
        Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
          differs between optimizer classes.
        * param_group - a dict containing all parameter groups.

        It almost copied from: https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.state_dict
        """

        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "_parameters"}
            param_mappings.update(
                {
                    id(p): i
                    for i, p in enumerate(group._parameters, start_index)
                    if id(p) not in param_mappings
                }
            )
            packed["params"] = [param_mappings[id(p)] for p in group._parameters]
            start_index += len(packed["params"])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {
            (param_mappings[id(k)] if isinstance(k, Tensor) else k): v
            for k, v in self._state.items()
        }
        return {
            "state": packed_state,
            "param_groups": param_groups,
        }

    def step(self, closure: Union[Callable, None] = None) -> Union[Tensor, None]:
        raise NotImplementedError()

    def clip_grad(self):
        r"""Clips gradient norm of an iterable of parameters. 
        The norm is computed over all gradients together, as if they were concatenated into a single vector.

        You can set the max_norm and norm_type. 

        For more details, you can refer to the documentation of each optimizer(like Adam, SGD and so on). 

        You can also refer the code in :func:`oneflow.nn.utils.clip_grad_norm_`

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
        """
        for param_group in self.param_groups:
            for param in param_group.parameters:
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.zeros_()

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

    @property
    def support_sparse(self):
        return False

    def _check_variables_in_graph(self, vars_conf):
        for param_group in self.param_groups:
            for param in param_group.parameters:
                if not param.requires_grad:
                    continue

                if param not in vars_conf:
                    raise ValueError(
                        f"Parameter <{param}> is not in the corresponding nn.Graph/nn.Module."
                        " Please make sure you call the module's to(..)/to_global(...) method first,"
                        " then add the module's parameters into an optimizer."
                    )

    def _check_variables_optimizer_bound(self, vars_conf):
        for param_group in self.param_groups:
            for param in param_group.parameters:
                if not param.requires_grad:
                    continue

                if vars_conf[param].bound_optimizer is None:
                    vars_conf[param].bound_optimizer = self
                elif vars_conf[param].bound_optimizer is not self:
                    raise ValueError(
                        f"<{vars_conf[param].name}> is already bound to another optimizer."
                    )

    def _generate_indexed_slices_optimizer_conf(self, job_conf, vars_conf):
        if not self.support_sparse:
            raise ValueError(f"{self.__class__} does not support sparse updating.")

        for param_group in self.param_groups:
            for param in param_group.parameters:
                if not param.requires_grad:
                    continue

                sparse_opt_conf = job_conf.mutable_indexed_slices_optimizer_conf()
                sparse_variable_op_names = sparse_opt_conf.mutable_include_op_names()
                sparse_variable_op_names.add_op_name(vars_conf[param].name)
