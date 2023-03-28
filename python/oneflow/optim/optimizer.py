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
from oneflow.nn.graph.proxy import ProxyTensor
from oneflow.nn.parameter import Parameter
from oneflow.nn.utils.clip_grad import clip_grad_norm_
import oneflow as flow
from collections import defaultdict, abc as container_abcs


class ContiguousParamsUnit(object):
    def __init__(self, bufsize, dtype, device):
        self.bufsize = bufsize
        self.index = (
            0  # record the next buf position will be used while copying parameters
        )
        self.param_buf = flow.zeros(bufsize, dtype=dtype, device=device)
        self.grad_buf = flow.zeros(bufsize, dtype=dtype, device=device)

    def __repr__(self) -> str:
        return str(
            {
                "bufsize": self.bufsize,
                "index": self.index,
                "param_buf": self.param_buf,
                "grad_buf": self.grad_buf,
            }
        )


class ParamGroup(object):
    def __init__(
        self, parameters: Dict[str, Any], default_options: Dict,
    ):
        # ParamGroup must be constructed by Dict["params": parameters: List[Parameter, Tensor or ProxyTensor], "...": ...]
        assert isinstance(parameters, dict) and "params" in parameters
        assert not isinstance(parameters["params"], (Parameter, Tensor))
        self._parameters = list()

        for p in parameters["params"]:
            if isinstance(p, (Parameter, Tensor)):
                self._parameters.append(p)
            elif isinstance(p, ProxyTensor):
                # Add parameter from nn.Graph
                self._parameters.append(p.to(Tensor))
            else:
                raise ValueError(
                    "parameters in ParamGroup must be Tensor or ProxyTensor."
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

        self._make_options_valid()
        self.contiguous_params = self._options.get("contiguous_params", False)
        if self.contiguous_params:
            self.params_dict = dict()
            self._contiguous_parameters = list()
            self._make_contiguous_params(parameters)

    def _make_contiguous_params(self, parameters):
        assert not any(
            [p.is_global for p in parameters["params"]]
        ), "All parameters must be local tensor for contiguous params."

        def numel_in_bucket(tensor: flow.Tensor):
            assert flow.is_floating_point(
                tensor
            ), "contiguous params should be float tensor"

            def align(x: int, unit_size: int):
                return (x + (unit_size - 1)) // unit_size * unit_size

            # tensor memory should be align to 512 bytes for cuda operations,
            # align size depends on floating type
            return align(
                tensor.numel(),
                flow._oneflow_internal.max_alignment_size()
                // (flow.finfo(p.dtype).bits // 8),
            )

        for p in parameters["params"]:
            buf_type = (p.dtype, p.device)
            self.params_dict[buf_type] = self.params_dict.get(
                buf_type, 0
            ) + numel_in_bucket(p)

        for (dtype, device), bufsize in self.params_dict.items():
            self.params_dict[(dtype, device)] = ContiguousParamsUnit(
                bufsize, dtype, device,
            )

        for p in parameters["params"]:
            buf_type = (p.dtype, p.device)
            size = p.numel()
            shape = p.data.shape
            index = self.params_dict[buf_type].index
            assert index + size <= self.params_dict[buf_type].bufsize

            self.params_dict[buf_type].param_buf[index : index + size] = (
                p.data.detach().clone().view(-1)
            )
            p.data = (
                self.params_dict[buf_type].param_buf[index : index + size].view(shape)
            )
            p.grad = (
                self.params_dict[buf_type].grad_buf[index : index + size].view(shape)
            )

            self.params_dict[buf_type].index += numel_in_bucket(p)

            """
            This empty_cache can reduce the memory fragments, but cannot
            release the origin parameters' memory.
            Contiguous parameters will use the double memory for parameters.
            """
            flow.cuda.empty_cache()

        for buf_type in self.params_dict.keys():
            self.params_dict[buf_type].param_buf.grad = self.params_dict[
                buf_type
            ].grad_buf
            self._contiguous_parameters.append(self.params_dict[buf_type].param_buf)

    def _make_options_valid(self):
        """handle the conflict between optimizer options
        """
        if self._options.get("contiguous_params", False) and self._options.get(
            "fused", False
        ):
            self._options["fused"] = False

            warnings.warn(
                "do not set contiguous_params and fused at the same time, "
                "now only contiguous_params is set."
            )

    def __getitem__(self, key):
        if key == "contiguous_params":
            return self._options.get("contiguous_params", False)
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

    def __repr__(self):
        res = self.options
        res["params"] = self.parameters
        return str(res)

    @property
    def options(self):
        return self._options

    @property
    def parameters(self):
        return self._parameters

    @property
    def contiguous_parameters(self):
        """return contiguous_parameters for fast updating
        """
        return self._contiguous_parameters


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


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class Optimizer(object):
    def __init__(self, parameters, options):
        self.param_groups = list()
        self.state = defaultdict(dict)
        self.defaults = options
        self.state["step"] = 0

        self._parse_input_parameters(parameters)

        self.step = _decorate_step(self.step)
        self._state_not_saved = [
            "_contiguous_parameters",
            "_parameters",
            "params_dict",
            "contiguous_params",
        ]

    def add_param_group(self, param_group) -> None:
        r"""
        
        Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.
        
        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        
        Example:

        >>> import oneflow
        >>> import oneflow.optim as optim
        >>> w1 = oneflow.ones(3, 3)
        >>> w1.requires_grad = True
        >>> w2 = oneflow.ones(3, 3)
        >>> w2.requires_grad = True
        >>> o = optim.SGD([w1])
        >>> o.param_groups[0]
        {'lr': 0.001, 'momentum': 0.0, 'dampening': 0.0, 'weight_decay': 0.0, 'nesterov': False, 'maximize': False, 'params': [tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]], dtype=oneflow.float32, requires_grad=True)]}
        >>> o.add_param_group({'params': w2})
        >>> o.param_groups[1]
        {'lr': 0.001, 'momentum': 0.0, 'dampening': 0.0, 'weight_decay': 0.0, 'nesterov': False, 'maximize': False, 'params': [tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]], dtype=oneflow.float32, requires_grad=True)]}

        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group["params"]
        if isinstance(params, flow.Tensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters need to be organized in ordered collections, but "
                "the ordering of tensors in sets will change between runs. Please use a list instead."
            )
        else:
            param_group["params"] = list(params)

        for param in param_group["params"]:
            if not isinstance(param, flow.Tensor):
                raise TypeError(
                    "optimizer can only optimize Tensors, "
                    "but one of the params is " + type(param)
                )
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "parameter group didn't specify a value of required optimization parameter "
                    + name
                )
            else:
                param_group.setdefault(name, default)
        params = param_group["params"]
        if len(params) != len(set(params)):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters; "
                "in future, this will cause an error; ",
                stacklevel=3,
            )

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group.parameters))

        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(ParamGroup(param_group, self.defaults))

        for param in param_group["params"]:
            assert param.is_leaf, "parameters must be leaf tensor"
            self.state[param] = dict()

    def load_state_dict(self, state_dict) -> None:
        r"""
        Load the state of the optimizer which is created by `state_dict` function.

        It almost copied from: https://pytorch.org/docs/1.10/_modules/torch/optim/optimizer.html#Optimizer.load_state_dict.
        """

        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of parameter groups"
            )

        for param, saved_param in zip(groups, saved_groups):
            # the contiguous_params property is remained in state_dict,
            # so contiguous_params of state_dict and current optimizer should match.
            if "contiguous_params" in param and param[
                "contiguous_params"
            ] != saved_param["_options"].get("contiguous_params", False):
                raise ValueError(
                    "loaded contiguous_params state doesn't match the optimizer"
                )

            if param["contiguous_params"]:
                param_list = param.contiguous_parameters
            else:
                param_list = param.parameters

            if len(param_list) != len(saved_param["params"]):
                raise ValueError(
                    "loaded state dict contains a parameter group "
                    "that doesn't match the size of optimizer's group"
                )

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable((g["params"] for g in saved_groups)),
                chain.from_iterable(
                    (
                        g.parameters
                        if not g["contiguous_params"]
                        else g.contiguous_parameters
                        for g in groups
                    )
                ),
            )
        }

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device or placement of param."""
            if isinstance(value, Tensor):
                if value.is_local:
                    value = value.to(param.device)
                else:
                    cpu_value_placement = flow.placement("cpu", value.placement.ranks)
                    cpu_param_placement = flow.placement("cpu", param.placement.ranks)
                    value = (
                        value.to_global(placement=cpu_value_placement)
                        .to_global(placement=cpu_param_placement, sbp=param.sbp)
                        .to_global(placement=param.placement)
                    )
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
        self.state = state

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            group._options = deepcopy(new_group["_options"])
            group._enable_clip_grad = new_group["_enable_clip_grad"]
            return group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.param_groups = param_groups

    def state_dict(self):
        r"""
        Returns the state of the optimizer as a :py:class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
          differs between optimizer classes.
        * param_group - a dict containing all parameter groups.

        It almost copied from: https://pytorch.org/docs/1.10/_modules/torch/optim/optimizer.html#Optimizer.state_dict.
        """

        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            if group["contiguous_params"]:
                param_list = group.contiguous_parameters
            else:
                param_list = group.parameters

            nonlocal start_index
            packed = {k: v for k, v in group.items() if k not in self._state_not_saved}
            param_mappings.update(
                {
                    id(p): i
                    for i, p in enumerate(param_list, start_index)
                    if id(p) not in param_mappings
                }
            )
            packed["params"] = [param_mappings[id(p)] for p in param_list]
            start_index += len(packed["params"])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {
            (param_mappings[id(k)] if isinstance(k, Tensor) else k): v
            for k, v in self.state.items()
        }
        return {
            "state": packed_state,
            "param_groups": param_groups,
        }

    def step(self, closure: Union[Callable, None] = None) -> Union[Tensor, None]:
        """Performs a single optimization step (parameter update).

        Args:
            closure (Union[Callable, None], optional): A closure that reevaluates the model and returns the loss. Optional for most optimizers.

        Returns:
            Union[Tensor, None]: The loss. 
        """
        raise NotImplementedError()

    def clip_grad(self, error_if_nonfinite: bool = False):
        r"""Clips gradient norm of an iterable of parameters. 
        The norm is computed over all gradients together, as if they were concatenated into a single vector.

        You can set the max_norm and norm_type. 

        For more details, you can refer to the documentation of each optimizer(like Adam, SGD and so on). 

        You can also refer the code in :func:`oneflow.nn.utils.clip_grad_norm_`

        Args:
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:``parameters`` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)

        """
        for param_group in self.param_groups:
            if param_group._enable_clip_grad:
                clip_grad_norm_(
                    param_group.parameters,
                    param_group["clip_grad_max_norm"],
                    param_group["clip_grad_norm_type"],
                    error_if_nonfinite,
                )
            else:
                warnings.warn(
                    "To enable clip_grad, passing the `clip_grad_max_norm` and `clip_grad_norm_type` parameters when instantializing the Optimizer."
                )

    def zero_grad(self, set_to_none: bool = False):
        """Sets the gradients of all optimized :class:`oneflow.Tensor` s to zero.

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
            if param_group["contiguous_params"]:
                param_list = param_group.contiguous_parameters
            else:
                param_list = param_group.parameters

            for param in param_list:
                param._zero_grad_(set_to_none)

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
                ParamGroup({"params": list(parameters)}, self.defaults)
            )
        elif isinstance(parameters, collections.abc.Iterable):
            # List[Dict]
            if isinstance(parameters[0], dict):
                for param in parameters:
                    assert isinstance(param, dict)
                    self.param_groups.append(ParamGroup(param, self.defaults))
            # List[Parameter or Tensor]
            else:
                self.param_groups.append(
                    ParamGroup({"params": parameters}, self.defaults)
                )
        else:
            raise TypeError(
                f"params argument given to the optimizer should be an iterable of Tensors or dicts, but got {type(parameters)}"
            )

    def _generate_grad_clip_conf_for_optim_conf(self, param_group, optimizer_conf):
        if not param_group._enable_clip_grad:
            return

        assert "clip_grad_max_norm" in param_group
        assert "clip_grad_norm_type" in param_group
        max_norm = float(param_group["clip_grad_max_norm"])
        norm_type = float(param_group["clip_grad_norm_type"])
        clip_grad_norm = optimizer_conf.clip_conf.clip_by_global_norm
        clip_grad_norm.max_norm = max_norm
        clip_grad_norm.norm_type = norm_type

    def _generate_lr_scale_for_optim_conf(self, param_group, optimizer_conf):
        if "lr_scale" not in param_group:
            return

        lr_scale = float(param_group["lr_scale"])
        optimizer_conf.lr_scale = lr_scale

    @property
    def support_sparse(self):
        """Whether the Optimizer support sparse update. 

        """
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

                sparse_opt_conf = job_conf.indexed_slices_optimizer_conf
                sparse_variable_op_names = sparse_opt_conf.include_op_names
                sparse_variable_op_names.op_name.append(vars_conf[param].name)
