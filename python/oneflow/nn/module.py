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
import warnings

import numpy as np
import oneflow as flow
from oneflow.framework.tensor import Tensor
from oneflow.nn.parameter import Parameter
from contextlib import contextmanager


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
    r"""Base class for all neural network modules.
    
    This class is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.Module.html.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        import oneflow.nn as nn
        import oneflow.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.

    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.

    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    """

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
        r"""
        add_module(name, module)
        
        Adds a child module to the current module.

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
        r"""
        register_buffer(name, tensor, persistent=True)
        
        Adds a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.
                
        Example::

            >>> self.register_buffer('running_mean', oneflow.zeros(num_features)) # doctest: +SKIP
        """
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
        r"""
        register_parameter(name, param)
        
        Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
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
        r"""
        parameters(recurse=True) -> Iterator[Parameter]
        
        Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> for param in model.parameters(): # doctest: +SKIP
            ...     print(type(param), param.size()) # doctest: +SKIP
            <class 'oneflow.Tensor'> oneflow.Size([10])

        """
        for (name, param) in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        r"""
        named_parameters(prefix="", recurse=True) -> Iterator[Tuple[str, Tensor]]
        
        Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters(): # doctest: +SKIP
            ...    if name in ['bias']: # doctest: +SKIP
            ...        print(param.size()) # doctest: +SKIP

        """
        gen = self._named_members(
            lambda module: module._parameters.items(), prefix=prefix, recurse=recurse
        )
        for elem in gen:
            yield elem

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        r"""
        buffers(recurse=True) -> Iterator[Tensor]
        
        Returns an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            oneflow.Tensor: module buffer

        Example::

            >>> for buf in model.buffers(): # doctest: +SKIP
            ...     print(type(buf), buf.size()) # doctest: +SKIP
            <class 'oneflow.Tensor'> oneflow.Size([10])

        """
        for (name, buf) in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        r"""
        named_buffers(prefix="", recurse=True) -> Iterator[Tuple[str, Tensor]]
        
        Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            (string, oneflow.Tensor): Tuple containing the name and buffer

        Example::

            >>> for name, buf in self.named_buffers(): # doctest: +SKIP
            ...    if name in ['running_var']: # doctest: +SKIP
            ...        print(buf.size()) # doctest: +SKIP

        """
        gen = self._named_members(
            lambda module: module._buffers.items(), prefix=prefix, recurse=recurse
        )
        for elem in gen:
            yield elem

    def children(self) -> Iterator["Module"]:
        r"""
        children() -> Iterator["Module"]
        
        Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
            
        Example::

            >>> import oneflow.nn as nn
            >>> l1 = nn.Linear(2, 2)
            >>> l2 = nn.Linear(2, 2)
            >>> net = nn.Sequential(l1, l2)
            >>> for idx, m in enumerate(net.children()):
            ...     print(idx, '->', m)
            0 -> Linear(in_features=2, out_features=2, bias=True)
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for (name, module) in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        r"""
        named_children() -> Iterator[Tuple[str, "Module"]]
        
        Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple containing a name and child module

        Example::

            >>> for name, module in model.named_children(): # doctest: +SKIP
            ...     if name in ['conv4', 'conv5']: # doctest: +SKIP
            ...         print(module) # doctest: +SKIP

        """
        memo = set()
        for (name, module) in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield (name, module)

    def modules(self) -> Iterator["Module"]:
        r"""
        modules() -> Iterator["Module"]
        
        Returns an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> import oneflow.nn as nn
            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            ...     print(idx, '->', m)
            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for (name, module) in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set["Module"]] = None, prefix: str = ""):
        r"""
        named_modules(memo=None, prefix="")
        
        Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> import oneflow.nn as nn
            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            ...     print(idx, '->', m)
            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """
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
        r"""
        train(mode=True)
        
        Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm1d`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self: T) -> T:
        r"""
        eval()
        
        Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm1d`,
        etc.

        This is equivalent with :meth:`self.train(False) <oneflow.nn.Module.train>`.

        Returns:
            Module: self
        """
        return self.train(False)

    def zero_grad(self, set_to_none: bool = False) -> None:
        r"""
        zero_grad(set_to_none=False)
        
        Sets gradients of all model parameters to zero. See similar function
        under :class:`oneflow.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`oneflow.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, "_is_replica", False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead."
            )

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

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
        r"""
        load_state_dict(state_dict, strict=True)
        
        Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~oneflow.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~oneflow.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
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
        r"""
        state_dict(destination=None, prefix="", keep_vars=False) -> Dict[str, Tensor]
        
        Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        Args:
            destination (dict, optional): Deprecated. This dict is returned
                with the module state saved in it. It should also have an
                attribute ``_metadata: dict`` to save metadata of the module
                state. If it's not provided, an ``OrderedDict`` is created and
                returned. Default: ``None``
            prefix (str, optional): a prefix added to parameter and buffer
                names to compose the keys in dict. Default: ``''``
            keep_vars (bool, optional): by default the :class:`~oneflow.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching is not performed. Default: ``False``

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> import oneflow.nn as nn
            >>> l1 = nn.Linear(2, 2)
            >>> l2 = nn.Linear(2, 2)
            >>> net = nn.Sequential(l1, l2)
            >>> net.state_dict().keys()
            odict_keys(['0.weight', '0.bias', '1.weight', '1.bias'])

        """
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
        r"""
        register_forward_pre_hook(hook)
        
        Registers a forward pre-hook on the module.

        The hook will be called every time before :func:`forward` is invoked.
        It should have the following signature::

            hook(module, input) -> None or modified input

        The input contains only the positional arguments given to the module.
        Keyword arguments won't be passed to the hooks and only to the ``forward``.
        The hook can modify the input. User can either return a tuple or a
        single modified value in the hook. We will wrap the value into a tuple
        if a single value is returned(unless that value is already a tuple).

        """
        self._forward_pre_hooks[len(self._forward_pre_hooks)] = hook

    def register_forward_hook(self, hook: Callable[..., None]) -> None:
        r"""
        register_forward_hook(hook)
        
        Registers a forward hook on the module.

        The hook will be called every time after :func:`forward` has computed an output.
        It should have the following signature::

            hook(module, input, output) -> None or modified output

        The input contains only the positional arguments given to the module.
        Keyword arguments won't be passed to the hooks and only to the ``forward``.
        The hook can modify the output. It can modify the input inplace but
        it will not have effect on forward since this is called after
        :func:`forward` is called.

        """
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
        r"""
        apply(fn)
        
        Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model.

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::
    
            >>> import oneflow as flow
            >>> import oneflow.nn as nn
            >>> @flow.no_grad()
            ... def init_weights(m):
            ...     print(m)
            ...     if type(m) == nn.Linear:
            ...         m.weight.fill_(1.0)
            ...         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            tensor([[1., 1.],
                    [1., 1.]], dtype=oneflow.float32, requires_grad=True)
            Linear(in_features=2, out_features=2, bias=True)
            tensor([[1., 1.],
                    [1., 1.]], dtype=oneflow.float32, requires_grad=True)
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def to(self, device: Optional[Union[str, flow.device]] = None):
        r"""
        to(device=None)
        
        Moves the parameters and buffers.

        Its signature is similar to :meth:`oneflow.Tensor.to`.
        The parameters and buffers will be moved :attr:`device`, if that is given.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`oneflow.device`): the desired device of the parameters
                and buffers in this module

        Returns:
            Module: self

        Examples::

            >>> import oneflow as flow
            >>> import oneflow.nn as nn
            >>> linear = nn.Linear(2, 2)
            >>> linear.weight.device
            device(type='cpu', index=0)
            >>> gpu1 = flow.device("cuda:1")
            >>> linear.to(gpu1)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight.device
            device(type='cuda', index=1)
            >>> cpu = flow.device("cpu")
            >>> linear.to(cpu)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight.device
            device(type='cpu', index=0)

        """

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
        r"""
        cpu()
        
        Moves all model parameters and buffers to the CPU.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cpu())

    def cuda(self: T, device: Optional[Union[int, flow.device]] = None) -> T:
        r"""
        cuda(device=None)
        
        Moves all model parameters and buffers to the GPU.

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
        r"""
        float()
        
        Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)

    def double(self: T) -> T:
        r"""
        double()
        
        Casts all floating point parameters and buffers to ``double`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)

    def half(self: T) -> T:
        r"""
        half()
        
        Casts all floating point parameters and buffers to ``half`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)

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
