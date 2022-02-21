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
import itertools
from collections import OrderedDict, namedtuple
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple, TypeVar, Union
import traceback

import numpy as np
import oneflow as flow
from oneflow.framework.tensor import Tensor
from oneflow.nn.parameter import Parameter


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    def __repr__(self):
        if not self.missing_keys and (not self.unexpected_keys):
            return "<All keys matched successfully>"
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [numSpaces * " " + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


T = TypeVar("T", bound="Module")


class Module(object):
    def __init__(self):
        self.training = True
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

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        for hook in itertools.chain(self._forward_pre_hooks.values()):
            result = hook(self, args)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                args = result

        res = self.forward(*args, **kwargs)

        for hook in itertools.chain(self._forward_hooks.values()):
            result = hook(self, args, res)
            if result is not None:
                res = result

        return res

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        """Adds a child module to the current module.

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
            raise TypeError("buffer name should be a string. Got {}".format(type(name)))
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and (not isinstance(tensor, Tensor)):
            raise TypeError(
                "cannot assign '{}' object to buffer '{}' (Tensor or None required)".format(
                    type(tensor), name
                )
            )
        else:
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )
        elif not isinstance(name, str):
            raise TypeError(
                "parameter name should be a string. Got {}".format(type(name))
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
                "cannot assign '{}' object to parameter '{}' (nn.Parameter or None required)".format(
                    type(param), name
                )
            )
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
                    "cannot assign '{}' as parameter '{}' (nn.Parameter or None expected)".format(
                        type(value), name
                    )
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
                        "cannot assign '{}' as child module '{}' (nn.Module or None expected)".format(
                            type(value), name
                        )
                    )
                modules[name] = value
            else:
                buffers = self.__dict__.get("_buffers")
                if buffers is not None and name in buffers:
                    if value is not None and (not isinstance(value, Tensor)):
                        raise TypeError(
                            "cannot assign '{}' as buffer '{}' (Tensor or None expected)".format(
                                type(value), name
                            )
                        )
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def _named_members(self, get_members_fn, prefix="", recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for (module_prefix, module) in modules:
            members = get_members_fn(module)
            for (k, v) in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield (name, v)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for (name, param) in self.named_parameters(recurse=recurse):
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
        for (name, buf) in self.named_buffers(recurse=recurse):
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
        for (name, module) in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        memo = set()
        for (name, module) in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield (name, module)

    def modules(self) -> Iterator["Module"]:
        for (name, module) in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set["Module"]] = None, prefix: str = ""):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield (prefix, self)
            for (name, module) in self._modules.items():
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
        for (name, param) in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param
        for (name, buf) in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf

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
            for (k, v) in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            self._parameters.items(), persistent_buffers.items()
        )
        local_state = {k: v for (k, v) in local_name_params if v is not None}
        for (name, param) in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if tuple(input_param.shape) != tuple(param.shape):
                    error_msgs.append(
                        "size mismatch for {}: copying a param with shape {} from checkpoint, the shape in current model is {}.".format(
                            key, input_param.shape, param.shape
                        )
                    )
                    continue
                try:
                    with flow.no_grad():
                        param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append(
                        'While copying the parameter "{}", an exception occurred : \n\n{}.'.format(
                            key,
                            "".join(
                                map(
                                    lambda line: "\t" + line,
                                    traceback.format_exc().splitlines(True),
                                )
                            ),
                        )
                    )
            elif strict:
                missing_keys.append(key)
        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix) :]
                    input_name = input_name.split(".", 1)[0]
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
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
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
            for (name, child) in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self)
        load = None
        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join(('"{}"'.format(k) for k in unexpected_keys))
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join(('"{}"'.format(k) for k in missing_keys))
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
        self._save_to_state_dict(destination, prefix, keep_vars)
        for (name, module) in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix)
            if hook_result is not None:
                destination = hook_result
        return destination

    def register_forward_pre_hook(self, hook: Callable[..., None]) -> None:
        self._forward_pre_hooks[len(self._forward_pre_hooks)] = hook

    def register_forward_hook(self, hook: Callable[..., None]) -> None:
        self._forward_hooks[len(self._forward_hooks)] = hook

    def _apply(self, fn, applied_dict=None):
        # A dict to store tensors that has already been applied.
        # There is no need to apply multiple times on a same tensor.
        if applied_dict is None:
            applied_dict = dict()

        for module in self.children():
            module._apply(fn, applied_dict)

        def can_use_assign_copy(tensor, tensor_applied):
            return tensor.is_local == tensor_applied.is_local

        for (key, param) in self._parameters.items():
            if param is None:
                continue

            need_apply = False
            if param not in applied_dict:
                need_apply = True
                assert isinstance(param, Parameter)
                assert param.is_leaf
                with flow.no_grad():
                    param_applied = fn(param)
                param_applied.requires_grad = param.requires_grad

                if param.grad is not None:
                    assert param.grad.is_leaf
                    with flow.no_grad():
                        grad_applied = fn(param.grad)
                    grad_applied.requires_grad = param.grad.requires_grad
                    param_applied.grad = grad_applied
            else:
                param_applied = applied_dict[param]

            if can_use_assign_copy(param_applied, param):
                if need_apply:
                    self._parameters[key].data = param_applied
                    applied_dict[param] = param_applied
                else:
                    # The parameter's data has already been set when it can use assign copy.
                    pass
            else:
                if need_apply:
                    new_param = Parameter(param_applied, param.requires_grad)
                    self._parameters[key] = new_param
                    applied_dict[param] = new_param
                else:
                    self._parameters[key] = applied_dict[param]

        for (key, buf) in self._buffers.items():
            if buf is not None:
                if buf not in applied_dict:
                    buf_applied = fn(buf)
                    self._buffers[key] = buf_applied
                    applied_dict[buf] = buf_applied
                else:
                    self._buffers[key] = applied_dict[buf]
        return self

    def apply(self: T, fn: Callable[["Module"], None]) -> T:
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def to(self, device: Optional[Union[str, flow.device]] = None):
        def convert(t):
            return t.to(device)

        return self._apply(convert)

    def to_consistent(self, *args, **kwargs):
        raise RuntimeError(
            ".to_consistent has been removed, please use .to_global instead"
        )

    def to_global(self, placement=None, sbp=None):
        def convert(t):
            return t.to_global(placement=placement, sbp=sbp)

        return self._apply(convert)

    def cpu(self: T) -> T:
        r"""Moves all model parameters and buffers to the CPU.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cpu())

    def cuda(self: T, device: Optional[Union[int, flow.device]] = None) -> T:
        r"""Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cuda(device))

    def float(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)

    def double(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``double`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        """Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ""

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for (key, module) in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines
        main_str = self._get_name() + "("
        if lines:
            if len(extra_lines) == 1 and (not child_lines):
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def _shallow_repr(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        lines = extra_lines
        main_str = self._get_name() + "("
        if lines:
            if len(extra_lines) == 1:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str
