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
import collections.abc
import inspect
import functools
import os

import torch as torch_original
import oneflow as flow_stable
import oneflow.experimental as flow
import numpy as np
from .generators import generator, random_tensor, Nothing


postulate = [".rand", ".Tensor"]


def torch_tensor_to_flow(x):
    return flow.tensor(x.cpu().numpy())


class PyTorchDoesNotSupportError(Exception):
    def __init__(self, exc):
        self.exc = exc

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"PyTorch error: {str(self.exc)}"


def get_args(callable, *args, **kwargs):
    try:
        spec = inspect.getfullargspec(callable)
        spec_args = spec.args
        if spec_args[0] == "self":
            del spec_args[0]
        for i, arg in enumerate(args):
            arg_name = spec_args[i]
            annotation = spec.annotations[arg_name]
            if isinstance(arg, generator):
                arg.to(annotation)
        for arg_name, arg in kwargs.items():
            annotation = spec.annotations[arg_name]
            if isinstance(arg, generator):
                arg.to(annotation)
    except:
        pass
    pytorch_args, pytorch_kwargs, oneflow_args, oneflow_kwargs = [], {}, [], {}

    def get_pytorch_value(x):
        if isinstance(x, DualObject):
            return x.pytorch
        return x

    def get_oneflow_value(x):
        if isinstance(x, DualObject):
            return x.oneflow
        return x

    def get_generator_value(x):
        if isinstance(x, generator):
            return x.value()
        return x

    for arg in args:
        arg = get_generator_value(arg)
        pytorch_args.append(get_pytorch_value(arg))
        oneflow_args.append(get_oneflow_value(arg))
    for key, value in kwargs.items():
        value = get_generator_value(value)
        if isinstance(value, Nothing):
            continue
        pytorch_kwargs[key] = get_pytorch_value(value)
        oneflow_kwargs[key] = get_oneflow_value(value)
    return pytorch_args, pytorch_kwargs, oneflow_args, oneflow_kwargs


counter = 0


def GetDualObject(name, pytorch, oneflow):
    global counter
    counter += 1

    skipped_magic_methods = [
        "__class__",
        "__mro__",
        "__new__",
        "__init__",
        "__getattr__",
        "__setattr__",
        "__getattribute__",
        "__dict__",
        "__weakref__",
        "__builtins__",
        "__qualname__",
        "__name__",
        "__str__",
        "__repr__",
    ]

    pytorch_methods = dir(pytorch)
    if hasattr(pytorch, "__call__") and "__call__" not in pytorch_methods:
        pytorch_methods.append("__call__")

    magic_methods_for_new_cls = {}

    for method_name in pytorch_methods:
        if method_name.startswith("__") and method_name not in skipped_magic_methods:
            # init a new 'method_name' variable other than the one in for loop,
            # avoid a pitfall:
            # https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
            def get_dual_method(method_name):
                # __call__ is special. We should not delegate the '__call__' of the torch wrapper of class 'nn.Conv2d'
                # to 'nn.Conv2d.__call__', as 'nn.Conv2d.__call__' belongs to the object of type 'nn.Conv2d'
                # (not the class itself)
                if method_name == "__call__":

                    def dual_method(self, *args, **kwargs):
                        (
                            pytorch_args,
                            pytorch_kwargs,
                            oneflow_args,
                            oneflow_kwargs,
                        ) = get_args(pytorch, *args, **kwargs)
                        # use () instead of '__call__'
                        try:
                            pytorch_res = pytorch(*pytorch_args, **pytorch_kwargs)
                        except Exception as e:
                            raise PyTorchDoesNotSupportError(e)
                        # only check if the method is a postulate when it is called
                        if name in postulate:
                            oneflow_res = torch_tensor_to_flow(pytorch_res)
                        else:
                            oneflow_res = oneflow(*oneflow_args, **oneflow_kwargs)
                        return GetDualObject("unused", pytorch_res, oneflow_res)

                else:

                    def dual_method(self, *args, **kwargs):
                        pytorch_method = getattr(pytorch, method_name)
                        oneflow_method = getattr(oneflow, method_name)
                        (
                            pytorch_args,
                            pytorch_kwargs,
                            oneflow_args,
                            oneflow_kwargs,
                        ) = get_args(pytorch_method, *args, **kwargs)
                        try:
                            pytorch_res = pytorch_method(
                                *pytorch_args, **pytorch_kwargs
                            )
                        except Exception as e:
                            raise PyTorchDoesNotSupportError(e)
                        oneflow_res = oneflow_method(*oneflow_args, **oneflow_kwargs)
                        return GetDualObject("unused", pytorch_res, oneflow_res)

                return dual_method

            magic_methods_for_new_cls[method_name] = get_dual_method(method_name)

    Cls = type(f"{name}_{counter}", (DualObject,), magic_methods_for_new_cls)
    return Cls(name, pytorch, oneflow)


class DualObject:
    def __init__(self, name, pytorch, oneflow):
        self.name = name
        self.pytorch = pytorch
        self.oneflow = oneflow

        if isinstance(pytorch, torch_original.nn.Module):
            state_dict = pytorch.state_dict()
            state_dict = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
            oneflow.load_state_dict(state_dict)
            dual_modules_to_test.append(self)

        if isinstance(pytorch, torch_original.Tensor):
            dual_objects_to_test.append(self)

    def __repr__(self):
        return f"PyTorch object:\n{self.pytorch}\n\nOneFlow object:\n{self.oneflow}"

    def __getattr__(self, key):
        pytorch_attr = getattr(self.pytorch, key)
        oneflow_attr = getattr(self.oneflow, key)
        new_name = f"{self.name}.{key}"

        return GetDualObject(new_name, pytorch_attr, oneflow_attr)


dual_modules_to_test = []
dual_objects_to_test = []
torch_type2checker = {}


def equality_checker(torch_type, flow_type):
    def deco(f):
        torch_type2checker[(torch_type, flow_type)] = f
        return f

    return deco


def check_equality(dual_object: DualObject):
    checker = torch_type2checker.get(
        (type(dual_object.pytorch), type(dual_object.oneflow)), None
    )
    if checker is None:
        for key, value in torch_type2checker.items():
            if isinstance(dual_object.pytorch, key[0]) and isinstance(
                dual_object.oneflow, key[1]
            ):
                checker = value
                break
    assert checker is not None
    return checker(dual_object.pytorch, dual_object.oneflow)


@equality_checker(torch_original.Tensor, flow.Tensor)
@equality_checker(torch_original.Tensor, flow_stable._oneflow_internal.Tensor)
def check_tensor_equality(torch_tensor, flow_tensor):
    # TODO: check dtype
    if torch_tensor.grad is not None:
        assert (
            flow_tensor.grad is not None
        ), "OneFlow tensor doesn't have grad while PyTorch tensor has one"
        if not np.allclose(
            torch_tensor.grad.detach().cpu().numpy(), flow_tensor.grad.numpy()
        ):
            return False
    return np.allclose(torch_tensor.detach().cpu().numpy(), flow_tensor.numpy())


def autotest(n=20, auto_backward=True, rtol=1e-4, atol=1e-5):
    verbose = os.getenv("ONEFLOW_TEST_VERBOSE") is not None

    def deco(f):
        @functools.wraps(f)
        def new_f(test_case):
            nonlocal n
            while n > 0:
                dual_modules_to_test.clear()
                dual_objects_to_test.clear()
                try:
                    res = f(test_case)
                except PyTorchDoesNotSupportError as e:
                    if verbose:
                        print(e)
                    continue
                # TODO: support types other than Tensor, like torch.Size/flow.Size
                if res is not None:
                    if not isinstance(res, collections.abc.Sequence):
                        res = [res]
                    for x in res:
                        if auto_backward:
                            if isinstance(x.pytorch, torch_original.Tensor):
                                x.sum().backward()
                        dual_objects_to_test.append(x)
                for x in dual_modules_to_test:
                    # x.state_dict().values() returns dual object with inconsistent values
                    for key in x.pytorch.state_dict().keys():
                        dual_objects_to_test.append(
                            GetDualObject(
                                "unused",
                                x.pytorch.state_dict()[key],
                                x.oneflow.state_dict()[key],
                            )
                        )
                for x in dual_objects_to_test:
                    test_case.assertTrue(check_equality(x))
                if verbose:
                    print("test passed")
                n -= 1

        return new_f

    return deco


def random_pytorch_tensor(
    ndim=None, dim0=1, dim1=None, dim2=None, dim3=None, dim4=None, requires_grad=True
):
    if isinstance(requires_grad, generator):
        requires_grad = requires_grad.value()
    pytorch_tensor = (
        random_tensor(ndim, dim0, dim1, dim2, dim3, dim4)
        .value()
        .requires_grad_(requires_grad)
    )
    flow_tensor = flow.tensor(pytorch_tensor.detach().cpu().numpy(), requires_grad=True)
    return GetDualObject("unused", pytorch_tensor, flow_tensor)


torch = GetDualObject("", torch_original, flow)


__all__ = ["torch", "autotest", "random_pytorch_tensor"]
