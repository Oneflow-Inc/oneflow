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
import builtins
from oneflow.framework.tensor import Tensor
from typing import overload, Tuple, Any

_device = flow.device
_bool = builtins.bool
_dtype = flow.dtype


class FunctionParameter:
    """
    This class is used to store params in single function signature. The object of 
    FunctionParameter is to record the properties of the param.
    
    This class is referenced from https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/python_arg_parser.h
    """

    def __init__(self, fmt, keyword_only):
        """
        The method is used to init function parameter.

        Args:
            fmt(string): a string describes the name of param and default value
            keyword_only(bool): If True, the param is keyword args in python
        """
        self.allow_none = False
        self.optional = False
        self.keyword_only = keyword_only
        self.allow_numbers_as_tensors = True
        self.default_scalar = 0
        type_str = fmt.split(" ")[0]
        if type_str.find("?") != -1:
            self.allow_none = True
            type_str = type_str[:-1]
        name_str = fmt.split(" ")[1]
        self.type = type_str

        if "=" in name_str:
            self.name = name_str.split("=")[0]
            self.optional = True
            self.set_default_str(name_str.split("=")[1])
        else:
            self.name = name_str

    def check(self, obj):
        # Check whether the object belongs to the corresponding type
        obj_type = self.type
        if obj_type == "Device":
            return isinstance(obj, (_device, str))
        elif obj_type == "bool":
            return isinstance(obj, _bool)
        elif obj_type == "Tensor":
            if isinstance(obj, Tensor):
                return True
            elif self.allow_numbers_as_tensors and isinstance(int, float):
                return True
            return False
        elif obj_type == "ScalarType":
            return isinstance(obj, (_dtype, float, int))
        elif obj_type == "MemoryFormat":
            return True
        else:
            return False

    def set_default_str(self, s):
        if s == "None":
            self.allow_none = True
        if self.type == bool:
            self.default_scalar = s == "True"


class FunctionSignature:
    """
    This class is used to parse single function signature.

    The class is referenced from https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/python_arg_parser.h
    """

    def __init__(self, fmt, index):
        """
        Args:
            fmt(string): single function signature
            index(int): the index of function signatures
        """
        self.params = list()
        self.index = index  # the index of this signature
        self.max_args = 0  # signature has max args
        self.min_args = (
            0  # the number of positional parameters of the parameter must be specified
        )
        self.max_pos_args = 0  # maximum number of positional parameters
        self.name = ""  # funtion name
        self._init_signature(fmt=fmt)

    def _init_signature(self, fmt):
        open_paren = fmt.find("(")
        if open_paren == -1:
            raise ValueError("missing opening parenthesis: " + fmt)
        self.name = fmt[:open_paren]

        keyword_only = False
        string_params = fmt[open_paren + 1 :].split(", ")
        # Instantiate each parameter of FunctionParameter
        for _, string_param in enumerate(string_params):
            if string_param == "*":
                keyword_only = True
            else:
                if string_param.endswith(")"):
                    string_param = string_param[:-1]
                self.params.append(FunctionParameter(string_param, keyword_only))

        self.max_args = len(self.params)
        for param in self.params:
            if not param.optional:
                self.min_args += 1
            if not param.keyword_only:
                self.max_pos_args += 1

    def parse(self, parsed_args, *args, **kwargs):
        nargs = len(args)
        remaining_kwargs = len(kwargs)
        arg_pos = 0
        # If the number of parameters passed in is greater than the
        # maximum number of functions signed by the function, return false
        if nargs > self.max_pos_args:
            return False
        i = 0
        for param in self.params:
            value = -1  # Use -1 to indicate that no value is passed
            is_kwd = False
            if arg_pos < nargs:
                if param.keyword_only:
                    # Keyword parameter appears in non-key parameter, this is false
                    return False
                value = args[arg_pos]
            elif kwargs:
                value = args[param.name]
                is_kwd = True
            if (
                not isinstance(value, Tensor)
                and value == -1
                and (param.optional or param.allow_none)
            ):
                parsed_args[i] = value
                i += 1
            elif not isinstance(value, Tensor) and value == -1:
                return False
            elif (isinstance(value, Tensor) or value != -1) and param.check(value):
                parsed_args[i] = value
                i += 1
            else:
                return False
            if not is_kwd:
                arg_pos += 1
            elif value != -1:
                remaining_kwargs -= 1
        if remaining_kwargs > 0:
            return False
        return True


class PythonArgParser:
    """
    This class is used to realize overload feature in C++. Different parameter list 
    forms are represented by fixed format strings.

    This class is referenced from https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/api/python/functional/python_arg_parser.cpp

    Args:
        fmts(List(string)): the template string for function args
    """

    def __init__(self, fmts):
        self.signatures_ = list()
        self.function_name = ""
        self.max_args = 0
        self.parsed_args = [0 for _ in range(5)]
        self._init_Parser(fmts)

    def _init_Parser(self, fmts):
        for i, fmt in enumerate(fmts):
            self.signatures_.append(FunctionSignature(fmt, i))
            i += 1
            self.max_args = max(self.max_args, self.signatures_[-1].max_args)
        if self.signatures_:
            self.function_name = self.signatures_[0].name

    def parse(self, *args, **kwargs):
        for _, signature in enumerate(self.signatures_):
            if signature.parse(self.parsed_args, False, *args, **kwargs):
                return self.parsed_args


@overload
def _parse_to(
    device: _device,
    dtype: _dtype,
    non_blocking: _bool,
    copy: _bool,
    *,
    memory_format: Any,
) -> Tuple[_device, _dtype, _bool, Any]:
    ...


@overload
def _parse_to(
    dtype: _dtype, non_blocking: _bool, copy: _bool, *, memory_format: Any
) -> Tuple[_device, _dtype, _bool, Any]:
    ...


@overload
def _parse_to(
    tensor: Tensor, non_blocking: _bool, copy: _bool, *, memory_format: Any
) -> Tuple[_device, _dtype, _bool, Any]:
    ...


def _parse_to(*args, **kwargs):
    r"""
    This function parses the type of passed args to return Tuple(_device, _bool, _bool, Any).

    This can be called as

        .. function:: _parse_to(device: _device, dtype: _dtype, non_blocking: _bool, copy: _bool, *,
              memory_format: Any)
           :noindex:

        .. function:: _parse_to(dtype: _dtype, non_blocking: _bool, copy: _bool, *,
              memory_format: Any)
           :noindex:

        .. function:: _parse_to(tensor: Tensor, non_blocking: _bool, copy: _bool, *,
              memory_format: Any)
           :noindex:

    Args:
        device(flow.device or str): the desired device of tensor. Default: None
        dtype(flow.dtype, optional): the desired data type of returned tensor. Default: None
        non_blocking(bool, optional): If True and the source is in pinned memory, the copy will 
        be asynchronous with respect to the host. Default: False
        copy(bool, optional): If True, the copy of tensor will be returned. Default: False
        memory_format():the desired memory format of tensor. Default: None
    
    Returns:
        Tuple(_device, _bool, _bool, Any)

    Note:
        Memory format is temporarily not supported, we just add this arg for matching pytorch API.
        In this method, we default return None for memory format.
    """
    # This formats is referenced from pytorch, using it to describe
    # the possible forms of parameter transfer.
    fmts = [
        "to(Device device=None, ScalarType dtype=None, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
        "to(ScalarType dtype, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
        "to(Tensor tensor, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    ]
    python_arg_parser = PythonArgParser(fmts=fmts)
    signatures = python_arg_parser.signatures_
    nargs, nkwargs = len(args), len(kwargs)
    if nargs + nkwargs > 5:
        raise ValueError(
            f"The max numbers of params is 5, but now we have {nargs+nkwargs}"
        )

    finalArgs = [None for _ in range(5)]
    for signature_ in signatures:
        if signature_.parse(finalArgs, *args, **kwargs):
            if signature_.index == 0:
                device = None if finalArgs[0] == -1 else flow.device(finalArgs[0])
                scalarType = None if finalArgs[1] == -1 else finalArgs[1]
                non_blocking = False if finalArgs[2] == -1 else finalArgs[2]
                memory_format = None
                return (device, scalarType, non_blocking, memory_format)
            elif signature_.index == 1:
                device = None
                scalarType = flow.float32 if finalArgs[0] == -1 else finalArgs[0]
                non_blocking = False if finalArgs[1] == -1 else finalArgs[1]
                memory_format = None
                return (device, scalarType, non_blocking, memory_format)
            elif signature_.index == 2:
                tensor_data = finalArgs[0]
                device = tensor_data.device
                scalarType = tensor_data.dtype
                non_blocking = False if finalArgs[1] == -1 else finalArgs[1]
                memory_format = None
                return (device, scalarType, non_blocking, memory_format)
        finalArgs = [None for _ in range(5)]
    raise ValueError("Don't have matched args.")
