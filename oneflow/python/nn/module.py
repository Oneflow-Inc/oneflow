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

from __future__ import absolute_import
from collections import OrderedDict, namedtuple
from typing import Union, TypeVar, Iterator, Optional, Set, Tuple, Dict, List, Callable
from inspect import signature
import itertools

import numpy as np

import oneflow as flow
import oneflow_api
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.ops import parallel_cast
from oneflow.python.framework.tensor import Tensor
from oneflow.python.ops.get_variable import api_get_variable as get_variable
from oneflow.python.ops import initializer_util


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__


# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar("T", bound="Module")


_counter = 0


def get_var_helper(shape, name=None):
    global _counter
    if name is None:
        name = "x_" + str(_counter)
    var = flow.get_variable(
        name, shape=shape, initializer=flow.kaiming_initializer(shape)
    )
    _counter += 1
    return var


@oneflow_export("nn.Parameter")
class Parameter(Tensor):
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        return getattr(self._data, name)


class InputConfigs:
    def __init__(self):
        self._configs = {}

    def __delitem__(self, key):
        del self._configs[key]

    def __getitem__(self, key):
        return self._configs[key]

    def __setitem__(self, key, value):
        self._configs[key] = value

    def _to_dict(self):
        return self._configs


# TODO: rename
@oneflow_export("nn.Module_v2")
class Module(object):
    def __init__(self):
        self.training = True
        self._consistent = False
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._non_persistent_buffers_set = set()
        self._backward_hooks = OrderedDict()
        self._is_full_backward_hook = None
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
        self.input_configs = InputConfigs()

    @property
    def consistent(self):
        return self._consistent

    def forward(self, *args):
        raise NotImplementedError()

    def consistent_forward(self, *args):
        return self.forward(*args)

    def force_mirrored_forward(self, *args):
        raise NotImplementedError()

    def __call__(self, *args):
        for hook in itertools.chain(self._forward_pre_hooks.values()):
            result = hook(self, args)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                args = result

        if self.consistent:
            is_force_mirrored_overloaded = (
                Module.__dict__["force_mirrored_forward"]
                != self.__class__.__dict__["force_mirrored_forward"]
            )
            if is_force_mirrored_overloaded:
                return self.force_mirrored_forward(*args)
            else:
                print(self.input_configs._to_dict())
                args = list(args)
                for key, value in self.input_configs._to_dict().items():
                    args[key] = parallel_cast(args[key], distribute=value)
                return self.consisten_forward(*args)
        else:
            return self.forward(*args)

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(type(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif "." in name:
            raise KeyError('module name can\'t contain ".", got: {}'.format(name))
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')
        self._modules[name] = module

    def register_buffer(
        self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(
                "buffer name should be a string. " "Got {}".format(type(name))
            )
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(
                "cannot assign '{}' object to buffer '{}' "
                "(Tensor or None required)".format(type(tensor), name)
            )
        else:
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional["Parameter"]) -> None:
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )
        elif not isinstance(name, str):
            raise TypeError(
                "parameter name should be a string. " "Got {}".format(type(name))
            )
        elif "." in name:
            raise KeyError('parameter name can\'t contain "."')
        elif name == "":
            raise KeyError('parameter name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                "cannot assign '{}' object to parameter '{}' "
                "(nn.Parameter or None required)".format(type(param), name)
            )
        # elif param.grad_fn:
        #     raise ValueError(
        #         "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
        #         "parameters must be created explicitly. To express '{0}' "
        #         "as a function of another Tensor, compute the value in "
        #         "the forward() method.".format(name))
        else:
            self._parameters[name] = param

    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, name)
        )

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call"
                )
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
            )
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    "cannot assign '{}' as parameter '{}' "
                    "(nn.Parameter or None expected)".format(type(value), name)
                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get("_modules")
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call"
                    )
                remove_from(
                    self.__dict__,
                    self._parameters,
                    self._buffers,
                    self._non_persistent_buffers_set,
                )
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        "cannot assign '{}' as child module '{}' "
                        "(nn.Module or None expected)".format(type(value), name)
                    )
                modules[name] = value
            else:
                buffers = self.__dict__.get("_buffers")
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, Tensor):
                        raise TypeError(
                            "cannot assign '{}' as buffer '{}' "
                            "(Tensor or None expected)".format(type(value), name)
                        )
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def _named_members(self, get_members_fn, prefix="", recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator["Parameter"]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        gen = self._named_members(
            lambda module: module._parameters.items(), prefix=prefix, recurse=recurse
        )
        for elem in gen:
            yield elem

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        gen = self._named_members(
            lambda module: module._buffers.items(), prefix=prefix, recurse=recurse
        )
        for elem in gen:
            yield elem

    def children(self) -> Iterator["Module"]:
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator["Module"]:
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set["Module"]] = None, prefix: str = ""):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def train(self: T, mode: bool = True) -> T:
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self: T) -> T:
        return self.train(False)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for name, param in self._parameters.items():
            if param is not None:
                # TODO: uncomment these line when tensor is ready
                # destination[prefix + name] = param if keep_vars else param.detach()
                destination[prefix + name] = param
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                # destination[prefix + name] = buf if keep_vars else buf.detach()
                destination[prefix + name] = buf

    # def load_state_dict(self, state_dict: Dict[str, np.ndarray], strict: bool = True):
    #     param_buffer_var_names = [x._get_var_name() for x in self.parameters()]
    #     state_dict_overlapped = {
    #         key: item
    #         for key, item in state_dict.items()
    #         if key in param_buffer_var_names
    #     }
    #     if strict and len(state_dict) != len(state_dict_overlapped):
    #         # TODO:
    #         raise RuntimeError()
    #     flow.load_variables(state_dict_overlapped)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for hook in self._load_state_dict_pre_hooks.values():
            hook(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            self._parameters.items(), persistent_buffers.items()
        )
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                if tuple(input_param.shape) != tuple(param.shape):
                    # local shape should match the one in checkpoint
                    error_msgs.append(
                        "size mismatch for {}: copying a param with shape {} from checkpoint, "
                        "the shape in current model is {}.".format(
                            key, input_param.shape, param.shape
                        )
                    )
                    continue
                try:
                    # with torch.no_grad():
                    # param.copy_(input_param)
                    flow.load_variables({param._variable_name: input_param})
                except Exception as ex:
                    error_msgs.append(
                        'While copying the parameter named "{}", '
                        "whose dimensions in the model are {} and "
                        "whose dimensions in the checkpoint are {}, "
                        "an exception occurred : {}.".format(
                            key, param.shape, input_param.shape, ex.args
                        )
                    )
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix) :]
                    input_name = input_name.split(".", 1)[
                        0
                    ]  # get the name of param/buffer/child
                    if (
                        input_name not in self._modules
                        and input_name not in local_state
                    ):
                        unexpected_keys.append(key)

    def load_state_dict(
        self,
        state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
        strict: bool = True,
    ):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self)
        load = None  # break load->load reference cycle

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in missing_keys)
                    ),
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def state_dict(
        self, destination=None, prefix="", keep_vars=False
    ) -> Dict[str, Tensor]:
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        # TODO:
        # destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)

        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            # hook_result = hook(self, destination, prefix, local_metadata)
            hook_result = hook(self, destination, prefix)
            if hook_result is not None:
                destination = hook_result
        return destination

    def register_forward_pre_hook(self, hook: Callable[..., None]) -> None:
        self._forward_pre_hooks[len(self._forward_pre_hooks)] = hook

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for key, param in self._parameters.items():
            if param is not None:
                assert isinstance(param, Parameter)
                assert param.is_leaf
                self._parameters[key] = Parameter(param_applied, param.requires_grad)

                if param.grad is not None:
                    assert param.grad.is_leaf
                    self._parameters[key].grad = grad_applied.requires_grad_(
                        param.grad.requires_grad
                    )

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def apply(self: T, fn: Callable[["Module"], None]) -> T:
        r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self
