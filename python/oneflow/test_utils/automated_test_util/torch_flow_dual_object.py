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
import functools
import inspect
import os
import warnings

import numpy as np
import oneflow as flow

try:
    import torch as torch_original
except ImportError:
    print(
        "automated_test_util module uses PyTorch to verify OneFlow module's interface and result. Please install Pytorch according `https://pytorch.org/get-started/locally/`."
    )

from .generators import Nothing, generator, random_tensor

postulate = [".rand", ".Tensor"]

testing = False
testing_graph = False


def torch_tensor_to_flow(x):
    return flow.tensor(x.cpu().numpy())


note_pytorch_method_names = []
note_pytorch_args = []
note_pytorch_kwargs = []
vis_tensor = []
vis_parameters = {}
call_tensor_id = []
extra_input_tensor = set()
eager_tensor_2_graph_tensor = dict()


class PyTorchDoesNotSupportError(Exception):
    def __init__(self, exc):
        self.exc = exc

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"PyTorch error: {str(self.exc)}"


class BothDoNotSupportError(Exception):
    def __init__(self, th_exc, of_exc):
        self.th_exc = th_exc
        self.of_exc = of_exc

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"PyTorch error: {str(self.th_exc)}\nOneFlow error: {str(self.of_exc)}"


call_pytorch = None


def get_tensor_shape(call_pytorch):
    shape_list = []
    for i in range(len(call_pytorch.shape)):
        shape_list.append(call_pytorch.shape[i])
    return shape_list


def get_args(callable, *args, **kwargs):
    try:
        spec = inspect.getfullargspec(callable)
        spec_args = spec.args
        if spec_args[0] == "self":
            del spec_args[0]
        for (i, arg) in enumerate(args):
            arg_name = spec_args[i]
            annotation = spec.annotations[arg_name]
            if isinstance(arg, generator):
                arg.to(annotation)
        for (arg_name, arg) in kwargs.items():
            annotation = spec.annotations[arg_name]
            if isinstance(arg, generator):
                arg.to(annotation)
    except:
        pass
    (pytorch_args, pytorch_kwargs, oneflow_args, oneflow_kwargs) = ([], {}, [], {})

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
        # TODO: refine codes
        if isinstance(arg, tuple):
            pytorch_tuple_args = []
            oneflow_tuple_args = []
            for t in arg:
                t = get_generator_value(t)
                pytorch_tuple_args.append(get_pytorch_value(t))
                oneflow_tuple_args.append(get_oneflow_value(t))
            pytorch_args.append(tuple(pytorch_tuple_args))
            oneflow_args.append(tuple(oneflow_tuple_args))
        else:
            arg = get_generator_value(arg)
            pytorch_args.append(get_pytorch_value(arg))
            oneflow_args.append(get_oneflow_value(arg))
    for (key, value) in kwargs.items():
        value = get_generator_value(value)
        if isinstance(value, Nothing):
            continue
        pytorch_kwargs[key] = get_pytorch_value(value)
        oneflow_kwargs[key] = get_oneflow_value(value)

    new_pytorch_args = []
    new_pytorch_kwargs = {}
    for x in pytorch_args:
        if isinstance(x, (tuple, list)):
            new_x = f"("
            len_x = len(x)
            for i in range(len_x):
                if type(x[i]) is torch_original.Tensor:
                    if i < len_x - 1:
                        new_x += f"Tensor({get_tensor_shape(x[i])}), "
                    else:
                        new_x += f"Tensor({get_tensor_shape(x[i])})"
                else:
                    if i < len_x - 1:
                        new_x += f"{x[i]}, "
                    else:
                        new_x += f"{x[i]}"
            new_x += f")"
            new_pytorch_args.append(new_x)
            continue
        if type(x) is torch_original.Tensor:
            new_pytorch_args.append(f"Tensor({get_tensor_shape(x)})")
        else:
            new_pytorch_args.append(x)
    for key, value in pytorch_kwargs.items():
        if type(value) is torch_original.Tensor:
            new_pytorch_kwargs[key] = f"Tensor({get_tensor_shape(value)})"
        else:
            new_pytorch_kwargs[key] = value

    if not isinstance(callable, (torch_original.nn.Module)):
        if isinstance(call_pytorch, torch_original.Tensor):
            note_pytorch_method_names.append(
                f"Tensor({get_tensor_shape(call_pytorch)}).{callable.__name__}"
            )
        elif isinstance(call_pytorch, torch_original.nn.Module):
            note_pytorch_method_names.append(f"Module.{callable.__name__}")
        else:
            note_pytorch_method_names.append(f"{callable.__name__}")
    else:
        note_pytorch_method_names.append(repr(callable))

    note_pytorch_args.append(new_pytorch_args)
    note_pytorch_kwargs.append(new_pytorch_kwargs)

    return (pytorch_args, pytorch_kwargs, oneflow_args, oneflow_kwargs)


counter = 0
align_exception = os.getenv("ONEFLOW_TEST_ALIGN_EXCEPTION") is not None


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

            def get_dual_method(method_name):
                if method_name == "__call__":

                    def dual_method(self, *args, **kwargs):
                        (
                            pytorch_args,
                            pytorch_kwargs,
                            oneflow_args,
                            oneflow_kwargs,
                        ) = get_args(pytorch, *args, **kwargs)

                        try:
                            pytorch_res = pytorch(*pytorch_args, **pytorch_kwargs)

                            if isinstance(pytorch_res, torch_original.Tensor):
                                if (
                                    hasattr(pytorch, "__name__")
                                    and pytorch.__name__ == "to"
                                    and (
                                        (
                                            len(pytorch_args) > 0
                                            and pytorch_args[0] == "cpu"
                                        )
                                        or (
                                            len(pytorch_kwargs) > 0
                                            and pytorch_kwargs["device"] == "cpu"
                                        )
                                    )
                                ):
                                    pass
                                elif (
                                    len(pytorch_args) > 0
                                    and isinstance(
                                        pytorch_args[0], torch_original.Tensor
                                    )
                                    and id(pytorch_args[0]) == id(pytorch_res)
                                ):
                                    extra_input_tensor.add(pytorch_res)
                                else:
                                    call_tensor_id.append(id(pytorch_res))

                        except Exception as e:
                            if align_exception:
                                try:
                                    oneflow_res = oneflow(
                                        *oneflow_args, **oneflow_kwargs
                                    )
                                except Exception as ee:
                                    raise BothDoNotSupportError(e, ee) from None
                                print(
                                    "PyTorch has an error but OneFlow is ok, maybe you should check your implementation to align with PyTorch."
                                )
                            raise PyTorchDoesNotSupportError(e)

                        if name in postulate:
                            oneflow_res = torch_tensor_to_flow(pytorch_res)
                        else:
                            oneflow_res = oneflow(*oneflow_args, **oneflow_kwargs)
                            if testing_graph:
                                find_check_module_func = True
                                ignore_apis_list = ["to", "tensor", "_to", "train"]
                                test_g_res = []
                                if isinstance(oneflow, flow.nn.Module):

                                    class TestGraphOfModule(flow.nn.Graph):
                                        def __init__(self):
                                            super().__init__()
                                            self.test_module = oneflow

                                        def build(self, *args):
                                            return self.test_module(*args)

                                    test_g = TestGraphOfModule()
                                    test_g_res = test_g(*oneflow_args)
                                elif oneflow.__name__ in ignore_apis_list:
                                    find_check_module_func = False
                                # 1. "oneflow.nn.modules" not in oneflow.__module__: For avoid run nn.Module branch graph test, like fold op call Fold Module actually.
                                # 2. inspect.isfunction(oneflow): Compared with the ordinary flow.xxx, oneflow.nn.modules.math_ops series op exist an extra layer of python wrapper.
                                # 3. inspect.ismethod(oneflow) and "oneflow.nn.modules" in oneflow.__module__:  For op that only has Tensor.xxx method, and call oneflow.xxx actually, like masked_fill.
                                elif (
                                    "oneflow.nn.modules" not in oneflow.__module__
                                    or inspect.isfunction(oneflow)
                                    or (
                                        inspect.ismethod(oneflow)
                                        and "oneflow.nn.modules" in oneflow.__module__
                                    )
                                ):

                                    class TestGraphOfFunctional(flow.nn.Graph):
                                        def __init__(self):
                                            super().__init__()
                                            self.test_module_func = oneflow

                                        def build(self):
                                            return self.test_module_func(
                                                *oneflow_args, **oneflow_kwargs
                                            )

                                    test_g = TestGraphOfFunctional()
                                    test_g_res = test_g()
                                if find_check_module_func:
                                    if isinstance(test_g_res, tuple):
                                        for idx, g_res in enumerate(test_g_res):
                                            eager_tensor_2_graph_tensor[
                                                oneflow_res[idx]
                                            ] = g_res
                                    else:
                                        eager_tensor_2_graph_tensor[
                                            oneflow_res
                                        ] = test_g_res

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
                            if isinstance(pytorch_res, torch_original.Tensor):
                                call_tensor_id.append(id(pytorch_res))
                        except Exception as e:
                            if align_exception:
                                try:
                                    oneflow_res = oneflow_method(
                                        *oneflow_args, **oneflow_kwargs
                                    )
                                except Exception as ee:
                                    raise BothDoNotSupportError(e, ee) from None
                                print(
                                    "PyTorch has an error but OneFlow is ok, maybe you should check your implementation to align with PyTorch."
                                )
                            raise PyTorchDoesNotSupportError(e)
                        oneflow_res = oneflow_method(*oneflow_args, **oneflow_kwargs)
                        if testing_graph:

                            class TestGraphOfTensorMethod(flow.nn.Graph):
                                def __init__(self):
                                    super().__init__()

                                def build(self):
                                    return oneflow_method(
                                        *oneflow_args, **oneflow_kwargs
                                    )

                            test_g = TestGraphOfTensorMethod()
                            test_g_res = test_g()
                            if isinstance(test_g_res, tuple):
                                for idx, g_res in enumerate(test_g_res):
                                    eager_tensor_2_graph_tensor[
                                        oneflow_res[idx]
                                    ] = g_res
                            else:
                                eager_tensor_2_graph_tensor[oneflow_res] = test_g_res

                        return GetDualObject("unused", pytorch_res, oneflow_res)

                return dual_method

            magic_methods_for_new_cls[method_name] = get_dual_method(method_name)
    Cls = type(f"{name}_{counter}", (DualObject,), magic_methods_for_new_cls)
    return Cls(name, pytorch, oneflow)


def note_print_args(x, end=True):
    if end:
        if isinstance(x, str) and "Tensor" not in x:
            print(f"\033[32m{x}, \033[0m", end="")
        else:
            print(f"\033[32m{x}, \033[0m", end="")
    else:
        if isinstance(x, str) and "Tensor" not in x:
            print(f"\033[32m{x}\033[0m", end="")
        else:
            print(f"\033[32m{x}\033[0m", end="")


def note_print_kwargs(x, y, end=True):
    if end:
        if isinstance(y, str) and "Tensor" not in y:
            print(f"\033[32m{x}={y}, \033[0m", end="")
        else:
            print(f"\033[32m{x}={y}, \033[0m", end="")
    else:
        if isinstance(y, str) and "Tensor" not in y:
            print(f"\033[32m{x}={y}\033[0m", end="")
        else:
            print(f"\033[32m{x}={y}\033[0m", end="")


def print_note_fake_program():
    code_len = len(note_pytorch_method_names)
    for i in range(code_len):
        note_pytorch_args_len = len(note_pytorch_args[i])
        note_pytorch_kwargs_len = len(note_pytorch_kwargs[i])
        print(f"\033[32m{note_pytorch_method_names[i]}\033[0m", end="")
        print(f"\033[32m(\033[0m", end="")
        if note_pytorch_args[i]:
            index = 0
            for x in note_pytorch_args[i]:
                index += 1
                note_print_args(x, index < note_pytorch_args_len)

        if note_pytorch_kwargs[i]:
            index = 0
            if note_pytorch_args[i]:
                print(f"\033[32m, \033[0m", end="")
            for x in note_pytorch_kwargs[i].keys():
                index += 1
                note_print_kwargs(
                    x, note_pytorch_kwargs[i][x], index < note_pytorch_kwargs_len
                )
        print(f"\033[32m)\033[0m")

    print(f"\033[32m-----------------------------------------------------------\033[0m")
    unique_vis_tensor = []
    flag_vis_tensor = [False for _ in range(len(vis_tensor))]
    for i in range(len(vis_tensor)):
        if flag_vis_tensor[i] == True:
            continue
        unique_vis_tensor.append(vis_tensor[i])
        flag_vis_tensor[i] = True
        for j in range(i + 1, len(vis_tensor)):
            if id(vis_tensor[i]) == id(vis_tensor[j]) and flag_vis_tensor[j] == False:
                flag_vis_tensor[j] = True

    if len(unique_vis_tensor) == 0:
        print(
            f"\033[32mThis program has {len(extra_input_tensor)} input tensor: \033[0m"
        )
        for input_tensor in iter(extra_input_tensor):
            print(f"\033[32mShape{get_tensor_shape(input_tensor)}\033[0m")
            print(f"\033[32m{input_tensor}\033[0m")
            print(
                f"\033[32m-----------------------------------------------------------\033[0m"
            )
    else:
        print(
            f"\033[32mThis program has {len(unique_vis_tensor)} input tensor: \033[0m"
        )
        for input_tensor in unique_vis_tensor:
            print(f"\033[32mShape{get_tensor_shape(input_tensor)}\033[0m")
            print(f"\033[32m{input_tensor}\033[0m")
            print(
                f"\033[32m-----------------------------------------------------------\033[0m"
            )
        if vis_parameters:
            print(
                f"\033[32m-------------------nn.Module Parameters---------------------\033[0m"
            )
            for name, param in vis_parameters.items():
                print(f"\033[32m{name}: {param}\033[0m")


def clear_note_fake_program():
    note_pytorch_method_names.clear()
    note_pytorch_args.clear()
    note_pytorch_kwargs.clear()
    call_tensor_id.clear()
    vis_tensor.clear()
    eager_tensor_2_graph_tensor.clear()
    vis_parameters.clear()
    extra_input_tensor.clear()


class DualObject:
    def __init__(self, name, pytorch, oneflow):
        self.name = name
        self.pytorch = pytorch
        self.oneflow = oneflow
        if isinstance(pytorch, torch_original.nn.Module):
            state_dict = pytorch.state_dict()
            state_dict = {k: v.detach().cpu().numpy() for (k, v) in state_dict.items()}
            oneflow.load_state_dict(state_dict, strict=False)
            if testing:
                dual_modules_to_test.append(self)
        if isinstance(pytorch, torch_original.Tensor):
            if testing:
                dual_objects_to_test.append(self)

    def __repr__(self):
        return f"PyTorch object:\n{self.pytorch}\n\nOneFlow object:\n{self.oneflow}"

    def __getattr__(self, key):
        pytorch_attr = getattr(self.pytorch, key)
        oneflow_attr = getattr(self.oneflow, key)
        new_name = f"{self.name}.{key}"
        global call_pytorch
        call_pytorch = self.pytorch
        return GetDualObject(new_name, pytorch_attr, oneflow_attr)


dual_modules_to_test = []
dual_objects_to_test = []
torch_type2checker = {}


def equality_checker(torch_type, flow_type):
    def deco(f):
        torch_type2checker[torch_type, flow_type] = f
        return f

    return deco


def check_equality(dual_object: DualObject, rtol=0.0001, atol=1e-05):
    checker = torch_type2checker.get(
        (type(dual_object.pytorch), type(dual_object.oneflow)), None
    )
    if checker is None:
        for (key, value) in torch_type2checker.items():
            if isinstance(dual_object.pytorch, key[0]) and isinstance(
                dual_object.oneflow, key[1]
            ):
                checker = value
                break
    assert checker is not None, (
        "checker not found for type "
        + str(type(dual_object.pytorch))
        + " and "
        + str(type(dual_object.oneflow))
    )
    return checker(dual_object.pytorch, dual_object.oneflow, rtol, atol)


@equality_checker(torch_original.Tensor, flow.Tensor)
@equality_checker(torch_original.Tensor, flow._oneflow_internal.Tensor)
def check_tensor_equality(torch_tensor, flow_tensor, rtol=0.0001, atol=1e-05):
    if torch_tensor.grad is not None:
        if flow_tensor.grad is None:
            print_note_fake_program()
        assert (
            flow_tensor.grad is not None
        ), f"OneFlow tensor doesn't have grad while PyTorch tensor has one, PyTorch tensor is\n {torch_tensor}\n, OneFlow tensor is\n{flow_tensor} "
        torch_grad = torch_tensor.grad.detach().cpu().numpy()
        flow_grad = flow_tensor.grad.numpy()
        if not np.allclose(
            torch_grad, flow_grad, rtol=rtol, atol=atol, equal_nan=True,
        ):
            print_note_fake_program()
            print("---------Grad Shape--------")
            print(torch_grad.shape)
            print(flow_grad.shape)
            print(
                f"Grads are not equal. PyTorch grad: \n{torch_grad}\n, OneFlow grad: \n{flow_grad}"
            )
            return False
    equality_res = np.allclose(
        torch_tensor.detach().cpu().numpy(),
        flow_tensor.numpy(),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    if equality_res == False:
        print_note_fake_program()
        print("---------Tensor Shape--------")
        print(torch_tensor.shape)
        print(flow_tensor.shape)
    return equality_res


@equality_checker(int, int)
@equality_checker(bool, bool)
def check_basetype_equality(a, b, ignored1, ignored2):
    return a == b


@equality_checker(type(None), type(None))
def check_nonetype_equality(a, b, ignored1, ignored2):
    return True


def autotest(
    n=20,
    auto_backward=True,
    rtol=0.0001,
    atol=1e-05,
    check_graph=True,
    check_allclose=True,
):
    verbose = os.getenv("ONEFLOW_TEST_VERBOSE") is not None

    def deco(f):
        @functools.wraps(f)
        def new_f(test_case):
            nonlocal n
            loop_limit = n * 20
            loop = 0
            while n > 0:
                clear_note_fake_program()
                if loop > loop_limit:
                    raise ValueError("autotest stuck in an endless loop!")
                dual_modules_to_test.clear()
                dual_objects_to_test.clear()
                try:
                    global testing
                    testing = True
                    global testing_graph
                    if check_graph:
                        testing_graph = True
                    res = f(test_case)
                    testing = False
                    testing_graph = False
                except (PyTorchDoesNotSupportError, BothDoNotSupportError) as e:
                    if verbose:
                        print(f"{f.__name__}")
                        print(e)
                    loop += 1
                    continue
                if res is not None:
                    if not isinstance(res, collections.abc.Sequence):
                        res = [res]
                    func_outputs = res
                    for x in res:
                        if auto_backward:
                            if isinstance(x.pytorch, torch_original.Tensor):
                                call_tensor_id.append(id(x.pytorch))
                                x.sum().backward()
                        dual_objects_to_test.append(x)
                for x in dual_modules_to_test:
                    for key in x.pytorch.state_dict().keys():
                        if key not in x.oneflow.state_dict().keys():
                            warnings.warn(f"oneflow module don't have `{key}`")
                            continue
                        vis_parameters[key] = x.pytorch.state_dict()[key]
                        dual_objects_to_test.append(
                            GetDualObject(
                                "unused",
                                getattr(x.pytorch, key),
                                getattr(x.oneflow, key),
                            )
                        )
                        call_tensor_id.append(id(getattr(x.pytorch, key)))
                        dual_objects_to_test.append(
                            GetDualObject(
                                "unused",
                                getattr(x.pytorch, key).grad,
                                getattr(x.oneflow, key).grad,
                            )
                        )
                        call_tensor_id.append(id(getattr(x.pytorch, key).grad))

                for x in dual_objects_to_test:
                    if (
                        isinstance(x.pytorch, torch_original.Tensor)
                        and id(x.pytorch) not in call_tensor_id
                    ):
                        vis_tensor.append(x.pytorch)
                # check eager
                for x in dual_objects_to_test:
                    if check_allclose:
                        test_case.assertTrue(check_equality(x, rtol=rtol, atol=atol), x)
                    if verbose:
                        print(f"{f.__name__} test eager passed.")
                # check graph
                for output in func_outputs:
                    flow_tensor = output.oneflow
                    if isinstance(flow_tensor, flow.Tensor):
                        if (
                            flow_tensor in eager_tensor_2_graph_tensor
                            and check_allclose
                        ):
                            test_case.assertTrue(
                                np.allclose(
                                    flow_tensor.numpy(),
                                    eager_tensor_2_graph_tensor[flow_tensor].numpy(),
                                    rtol=rtol,
                                    atol=atol,
                                    equal_nan=True,
                                )
                            )
                            if verbose:
                                print(f"{f.__name__} test graph passed.")
                        else:
                            if check_graph and check_allclose:
                                test_case.assertTrue(
                                    False,
                                    f"{f.__name__} cannot find module to check graph.",
                                )
                    else:
                        warnings.warn(
                            f"some outputs of {f.__name__} fail to check graph."
                        )
                n -= 1
                loop += 1

        return new_f

    return deco


def random_pytorch_tensor(
    ndim=None,
    dim0=1,
    dim1=None,
    dim2=None,
    dim3=None,
    dim4=None,
    low=0,
    high=1,
    dtype=float,
    requires_grad=True,
):
    if isinstance(requires_grad, generator):
        requires_grad = requires_grad.value()
    pytorch_tensor = (
        random_tensor(ndim, dim0, dim1, dim2, dim3, dim4, low, high, dtype)
        .value()
        .requires_grad_(requires_grad and dtype != int)
    )
    flow_tensor = flow.tensor(
        pytorch_tensor.detach().cpu().numpy(),
        requires_grad=(requires_grad and dtype != int),
    )
    return GetDualObject("unused", pytorch_tensor, flow_tensor)


torch = GetDualObject("", torch_original, flow)
__all__ = ["autotest", "random_pytorch_tensor"]
