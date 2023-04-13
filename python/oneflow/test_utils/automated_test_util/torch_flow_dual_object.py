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
import copy
import os
import warnings
import gc
from typing import Union

import numpy as np
import oneflow as flow
from oneflow.test_utils.automated_test_util import profiler as auto_profiler
from oneflow.test_utils.test_util import type_name_to_flow_type

flow.backends.cudnn.deterministic = True

try:
    import torch as torch_original

    torch_original.backends.cudnn.deterministic = True
    torch_original.set_printoptions(profile="full")
except ImportError:
    print(
        "automated_test_util module uses PyTorch to verify OneFlow module's interface and result. Please install Pytorch according `https://pytorch.org/get-started/locally/`."
    )


from .util import broadcast
from .global_scope import *
from .generators import (
    Nothing,
    generator,
    random_pytorch_tensor,
    random_pytorch_dtype,
    choice_pytorch_tensor,
    rng,
)

postulate = [".rand", ".Tensor"]

testing = False
testing_graph = False
global_check_allclose = True
global_atol = 1e-5
global_rtol = 1e-5
global_backward = True


def torch_tensor_to_flow(x):
    return flow.tensor(x.cpu().numpy())


note_pytorch_method_names = []
note_pytorch_args = []
note_pytorch_kwargs = []
vis_tensor = []
vis_parameters = {}
call_tensor_id = []
extra_input_tensor = []


class PyTorchDoesNotSupportError(Exception):
    def __init__(self, exc):
        self.exc = exc

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"PyTorch error: {str(self.exc)}"


class OneFlowGraphBuildOrRunError(Exception):
    def __init__(self, exc):
        self.exc = exc

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"OneFlow nn.Graph Build Or Run Error: {str(self.exc)}"


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
        if isinstance(arg, (tuple, list)):
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


def to_string(*args, **kwargs) -> str:
    def _to_string(x):
        if isinstance(x, DualObject):
            return x.name
        return str(x)

    strs = []
    if len(args) > 0:
        strs.append(", ".join([_to_string(arg) for arg in args]))
    if len(kwargs) > 0:
        strs.append(", ".join([f"{k}={_to_string(v)}" for k, v in kwargs.items()]))
    return ", ".join(strs)


counter = 0
align_exception = os.getenv("ONEFLOW_TEST_ALIGN_EXCEPTION") is not None


def check_eager_graph_tensor(eager_res, graph_res):
    if (
        global_check_allclose
        and isinstance(eager_res, flow.Tensor)
        and isinstance(graph_res, flow.Tensor)
    ):
        equality_res = np.allclose(
            eager_res.numpy(),
            graph_res.numpy(),
            rtol=global_rtol,
            atol=global_atol,
            equal_nan=True,
        )
        return equality_res
    else:
        return True


# NOTE(lixiang): Deepcopy the input parameters in order to correctly test the inplace version of the op.
def get_args_copy(args, kwargs):
    copy_args = []
    for arg in args:
        if flow.is_tensor(arg):
            copy_arg = arg.clone().detach()
        else:
            copy_arg = copy.deepcopy(arg)
        copy_args.append(copy_arg)
    copy_kwargs = {}
    for key, value in kwargs.items():
        if flow.is_tensor(value):
            copy_kwargs[key] = value.clone().detach()
        else:
            copy_kwargs[key] = copy.deepcopy(value)
    return copy_args, copy_kwargs


def get_fake_program_more_detail(oneflow, mode, func, args=None, kwargs=None):
    print(f"\033[1;33m============= {mode} ================\033[1;33m")
    print(f"\033[1;33mEnter {func} function\033[1;33m")
    try:
        if "__self__" in dir(oneflow) and flow.is_tensor(oneflow.__self__):
            print(f"\033[1;33m{oneflow.__self__}\033[1;33m")
    except:
        if flow.is_tensor(oneflow):
            print(f"\033[1;33m{oneflow}\033[1;33m")
    if args is not None:
        print(f"\033[1;33m{args}\033[1;33m")
    if kwargs is not None:
        print(f"\033[1;33m{kwargs}\033[1;33m")
    print_note_fake_program()
    print(f"\033[1;33mLeave {func} function\033[1;33m")
    print(f"\033[1;37m\033[1;37m")
    print("\n\n")


# NOTE(lixiang): When the graph global test is executed, the func is used to get the device type.
def get_global_test_device(oneflow_args, oneflow_kwargs=None):
    # The case when the parameter input of Op only has kwargs.
    if not oneflow_args:
        return oneflow_kwargs["placement"].type
    # The case when the parameter input of Op is tensors.
    elif isinstance(oneflow_args[0], flow.Tensor):
        return oneflow_args[0].placement.type
    # The case when the parameter input of Op is tensor.
    elif isinstance(oneflow_args[0], flow.placement):
        return oneflow_args[0].type
    # The case when the parameter input of Op is tuple. For example: test_0_dim_tensor.
    elif isinstance(oneflow_args[0], tuple):
        return oneflow_args[0][0].placement.type
    # When oneflow_args[0] is int or float, etc.
    else:
        return oneflow_args[1].placement.type


# NOTE(lixiang): When oneflow is of type nn.Module, build the following Graph for testing.
#   graph_train_oneflow: is a deepcopy of oneflow.
def get_module_graph_test(graph_train_oneflow, oneflow, verbose, oneflow_args, *args):
    of_sgd = flow.optim.SGD(graph_train_oneflow.parameters(), lr=0.001, momentum=0.9,)
    graph_train_parameters_len = 0
    for param in oneflow._parameters.values():
        if param is not None:
            graph_train_parameters_len += 1

    if verbose:
        get_fake_program_more_detail(
            oneflow, "nn.Graph", "get_module_graph_test", oneflow_args
        )

    class TestGraphOfModule(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.test_module = graph_train_oneflow
            if global_backward and graph_train_parameters_len:
                self.add_optimizer(of_sgd)

        def build(self, *args):
            res = self.test_module(*args)
            forward_res = res
            if global_backward and graph_train_parameters_len:
                if isinstance(self.test_module.to(flow.nn.Module), flow.nn.LSTMCell):
                    res = res[0] + res[1]
                elif isinstance(self.test_module.to(flow.nn.Module), flow.nn.LSTM):
                    res = res[0].sum() + res[1][0].sum() + res[1][1].sum()
                elif isinstance(res, (tuple, list)):
                    res = res[0]
                res = res.sum()
                res.backward()
            return forward_res

    try:
        test_g_res = TestGraphOfModule()
    except Exception as e:
        if not verbose:
            get_fake_program_more_detail(
                oneflow, "nn.Graph", "get_module_graph_test", oneflow_args
            )
        raise OneFlowGraphBuildOrRunError(e)
    return test_g_res


def check_oneflow_args_first_element_is_int(args):
    if isinstance(args, (tuple, list)) and len(args) > 0:
        if isinstance(args[0], (int, float)):
            return True
        elif isinstance(args[0], (tuple, list)):
            return check_oneflow_args_first_element_is_int(args[0])
    return False


# NOTE(lixiang): When oneflow is of functional type, build the following Graph for testing, and return the test results in Graph mode.
#   graph_functional_oneflow: is a deepcopy of oneflow.
def get_functional_graph_res(
    graph_functional_oneflow,
    oneflow,
    oneflow_res,
    oneflow_args,
    oneflow_kwargs,
    verbose,
    *graph_args,
    **graph_kwargs,
):
    test_g_res = []

    if verbose:
        get_fake_program_more_detail(
            oneflow,
            "nn.Graph",
            "get_functional_graph_res",
            oneflow_args,
            oneflow_kwargs,
        )

    class TestGraphOfFunctional(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self):
            return graph_functional_oneflow(*graph_args, **graph_kwargs)

    try:
        is_global_flag = is_global()

        # In graph mode, when the tensor on the cpu executes the to("cpu") method, a check error will be reported.
        if oneflow.__name__ == "to" or oneflow.__name__ == "_to":
            if isinstance(oneflow_res, flow.Tensor):
                # The global tensor needs to obtain the device type through placement.type.
                if is_global_flag:
                    if (
                        oneflow_args and oneflow_res.placement.type == oneflow_args[0]
                    ) or (
                        oneflow_kwargs
                        and oneflow_res.placement.type == oneflow_kwargs["device"]
                    ):
                        test_g_res = oneflow_res
                # The tensor needs to obtain the device type through device.type.
                else:
                    if (
                        oneflow_args and oneflow_res.device.type == oneflow_args[0]
                    ) or (
                        oneflow_kwargs
                        and oneflow_res.device.type == oneflow_kwargs["device"]
                    ):
                        test_g_res = oneflow_res
            else:
                pass
        # nn.Graph donot deal with Module type. EX: m.to_global(placement, sbp).
        elif oneflow.__name__ == "to_global":
            test_g_res = oneflow_res
        elif oneflow.__name__ == "Parameter":
            # nn.Graph donot deal with Parameter creation.
            test_g_res = oneflow_res
        # oneflow_args may be empty, such as dropout.
        elif is_global_flag and len(oneflow_args) == 0:
            test_g_res = oneflow_res
        # For some ops whose input parameters is int, 'int' object has no attribute 'placement'.
        elif (
            is_global_flag
            and len(oneflow_args) != 0
            and (check_oneflow_args_first_element_is_int(oneflow_args))
        ):
            test_g_res = oneflow_res
        # When doing the global op test, get_global_test_device() will be executed, and temporarily skipping the graph autotest on cpu device.
        elif (
            is_global_flag
            and oneflow.__name__ != "weight_norm"
            and (get_global_test_device(oneflow_args, oneflow_kwargs) == "cpu")
        ):
            test_g_res = oneflow_res
        else:
            test_g = TestGraphOfFunctional()
            test_g_res = test_g()
    except Exception as e:
        if not verbose:
            get_fake_program_more_detail(
                oneflow,
                "nn.Graph",
                "get_functional_graph_res",
                oneflow_args,
                oneflow_kwargs,
            )
        raise OneFlowGraphBuildOrRunError(e)
    return test_g_res


# NOTE(lixiang): When oneflow is of tensor type, build the following Graph for testing, and return the test results in Graph mode.
#   graph_tensor_oneflow is a deepcopy of oneflow.
def get_tensor_graph_res(
    graph_tensor_oneflow, oneflow, verbose, *tensor_graph_args, **tensor_graph_kwargs
):
    test_g_res = []

    if verbose:
        get_fake_program_more_detail(
            oneflow,
            "nn.Graph",
            "get_tensor_graph_res",
            tensor_graph_args,
            tensor_graph_kwargs,
        )

    class TestGraphOfTensorMethod(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self):
            return graph_tensor_oneflow(*tensor_graph_args, **tensor_graph_kwargs)

    try:
        # Set test_g_res = None, check_eager_graph_tensor will return True, the purpose is to temporarily skip the Graph global test on cpu.
        if is_global() and (get_global_test_device((oneflow,)) == "cpu"):
            test_g_res = None
        else:
            test_g = TestGraphOfTensorMethod()
            test_g_res = test_g()
    except Exception as e:
        if not verbose:
            get_fake_program_more_detail(
                oneflow,
                "nn.Graph",
                "get_tensor_graph_res",
                tensor_graph_args,
                tensor_graph_kwargs,
            )
        raise OneFlowGraphBuildOrRunError(e)
    return test_g_res


def get_oneflow_eager_res(
    oneflow, oneflow_args, oneflow_kwargs, verbose, is_tesnor_method=False
):
    if verbose:
        get_fake_program_more_detail(
            oneflow, "Eager", "get_oneflow_eager_res", oneflow_args, oneflow_kwargs
        )
    if not is_tesnor_method:
        oneflow_res = oneflow(*oneflow_args, **oneflow_kwargs)
    else:
        oneflow_res = oneflow(*oneflow_args, **oneflow_kwargs)
    return oneflow_res


# NOTE(lixiang): Check if the results of eager and graph are equal when oneflow is of type nn.Module or functional.
def oneflow_eager_run_with_graph_check(
    oneflow, oneflow_args, oneflow_kwargs, testing_graph, verbose, *args
):
    if testing_graph:
        graph_args, graph_kwargs = get_args_copy(oneflow_args, oneflow_kwargs)

        if isinstance(oneflow, flow.nn.Module):
            graph_train_oneflow = copy.deepcopy(oneflow)
            if not is_global():
                arg_device_type = "cpu"
                for arg in oneflow_args:
                    if flow.is_tensor(arg):
                        arg_device_type = arg.device.type
                graph_train_oneflow = graph_train_oneflow.to(arg_device_type)

        else:
            graph_functional_oneflow = copy.deepcopy(oneflow)

    oneflow_res = get_oneflow_eager_res(oneflow, oneflow_args, oneflow_kwargs, verbose)
    if testing_graph:
        find_check_module_func = True
        ignore_apis_list = ["tensor", "train"]
        test_g_res = []
        if isinstance(oneflow, flow.nn.Module):
            test_g = get_module_graph_test(
                graph_train_oneflow, oneflow, verbose, oneflow_args, *args
            )
            # When doing the global op test, get_global_test_device() will be executed, and temporarily skipping the graph autotest on cpu device.
            if is_global() and (
                get_global_test_device(oneflow_args, oneflow_kwargs) == "cpu"
            ):
                test_g_res = oneflow_res
            else:
                # When testing module methods, kwargs are not considered.
                test_g_res = test_g(*graph_args)
        elif oneflow.__name__ in ignore_apis_list:
            find_check_module_func = False
        # 1. "oneflow.nn.modules" not in oneflow.__module__: For avoid run nn.Module branch graph test, like fold op call Fold Module actually.
        # 2. inspect.isfunction(oneflow): Compared with the ordinary flow.xxx, oneflow.nn.modules.math_ops series op exist an extra layer of python wrapper.
        # 3. inspect.ismethod(oneflow) and "oneflow.nn.modules" in oneflow.__module__:  For op that only has Tensor.xxx method, and call oneflow.xxx actually, like masked_fill.
        elif (
            (
                oneflow.__module__ is not None
                and ("oneflow.nn.modules" not in oneflow.__module__)
            )
            or inspect.isfunction(oneflow)
            or (
                inspect.ismethod(oneflow) and "oneflow.nn.modules" in oneflow.__module__
            )
        ):

            test_g_res = get_functional_graph_res(
                graph_functional_oneflow,
                oneflow,
                oneflow_res,
                oneflow_args,
                oneflow_kwargs,
                verbose,
                *graph_args,
                **graph_kwargs,
            )
        if find_check_module_func:
            if isinstance(test_g_res, tuple):
                for _, g_res in enumerate(test_g_res):
                    if not check_eager_graph_tensor(oneflow_res, g_res):
                        get_fake_program_more_detail(
                            oneflow,
                            "Eager + nn.Graph",
                            "oneflow_eager_run_with_graph_check",
                            oneflow_args,
                            oneflow_kwargs,
                        )
            else:
                if not check_eager_graph_tensor(oneflow_res, test_g_res):
                    get_fake_program_more_detail(
                        oneflow,
                        "Eager + nn.Graph",
                        "oneflow_eager_run_with_graph_check",
                        oneflow_args,
                        oneflow_kwargs,
                    )
    return oneflow_res


# NOTE(lixiang): Check if the results of eager and graph are equal when oneflow is of type tensor.
def oneflow_tensor_eager_run_with_graph_check(
    oneflow, oneflow_method, oneflow_args, oneflow_kwargs, testing_graph, verbose
):
    if testing_graph:
        tensor_graph_args, tensor_graph_kwargs = get_args_copy(
            oneflow_args, oneflow_kwargs
        )
        graph_tensor_oneflow = copy.deepcopy(oneflow_method)

    oneflow_res = get_oneflow_eager_res(
        oneflow_method, oneflow_args, oneflow_kwargs, verbose, is_tesnor_method=True
    )

    if testing_graph:

        test_g_res = get_tensor_graph_res(
            graph_tensor_oneflow,
            oneflow,
            verbose,
            *tensor_graph_args,
            **tensor_graph_kwargs,
        )

        if isinstance(test_g_res, tuple):
            for _, g_res in enumerate(test_g_res):
                if not check_eager_graph_tensor(oneflow_res, g_res):
                    get_fake_program_more_detail(
                        oneflow,
                        "nn.Graph",
                        "oneflow_tensor_eager_run_with_graph_check",
                        oneflow_args,
                        oneflow_kwargs,
                    )
        else:
            if not check_eager_graph_tensor(oneflow_res, test_g_res):
                get_fake_program_more_detail(
                    oneflow,
                    "nn.Graph",
                    "oneflow_tensor_eager_run_with_graph_check",
                    oneflow_args,
                    oneflow_kwargs,
                )
    return oneflow_res


def get_pytorch_oneflow_res(
    pytorch,
    oneflow,
    pytorch_args,
    pytorch_kwargs,
    oneflow_args,
    oneflow_kwargs,
    name,
    verbose,
    testing_graph,
    *args,
):
    try:
        pytorch_res = pytorch(*pytorch_args, **pytorch_kwargs)

        if isinstance(pytorch_res, torch_original.Tensor):
            call_flag = True
            source_flag = True
            for x in pytorch_args:
                if isinstance(x, (tuple, list)):
                    for y in x:
                        if torch_original.is_tensor(y):
                            source_flag = False
                            if (
                                id(pytorch_res) == id(y)
                                and pytorch_res.device.type == y.device.type
                            ):
                                call_flag = False
                                break
                elif torch_original.is_tensor(x):
                    source_flag = False
                    if (
                        id(pytorch_res) == id(x)
                        and pytorch_res.device.type == x.device.type
                    ):
                        call_flag = False
                        break
            for x in pytorch_kwargs.values():
                if isinstance(x, (tuple, list)):
                    for y in x:
                        if torch_original.is_tensor(y):
                            source_flag = False
                            if (
                                id(pytorch_res) == id(y)
                                and pytorch_res.device.type == y.device.type
                            ):
                                call_flag = False
                                break
                elif torch_original.is_tensor(x):
                    source_flag = False
                    if (
                        id(pytorch_res) == id(x)
                        and pytorch_res.device.type == x.device.type
                    ):
                        call_flag = False
                        break
            if source_flag and pytorch.__name__ != "to":
                call_tensor_id.append(id(pytorch_res))
                extra_input_tensor.append(pytorch_res)
            elif call_flag:
                call_tensor_id.append(id(pytorch_res))

    except Exception as e:
        if align_exception:
            try:
                oneflow_res = oneflow(*oneflow_args, **oneflow_kwargs)
            except Exception as ee:
                raise BothDoNotSupportError(e, ee) from None
            print(
                "PyTorch has an error but OneFlow is ok, maybe you should check your implementation to align with PyTorch."
            )
            get_fake_program_more_detail(
                oneflow,
                "Eager",
                "get_pytorch_oneflow_res",
                oneflow_args,
                oneflow_kwargs,
            )
        raise PyTorchDoesNotSupportError(e)

    if name in postulate:
        oneflow_res = torch_tensor_to_flow(pytorch_res)
    else:
        oneflow_res = oneflow_eager_run_with_graph_check(
            oneflow, oneflow_args, oneflow_kwargs, testing_graph, verbose, *args,
        )
    return pytorch_res, oneflow_res


def get_pytorch_oneflow_tensor_res(
    pytorch_method,
    oneflow_method,
    oneflow,
    pytorch_args,
    pytorch_kwargs,
    oneflow_args,
    oneflow_kwargs,
    testing_graph,
    verbose,
):
    try:
        pytorch_res = pytorch_method(*pytorch_args, **pytorch_kwargs)
        if isinstance(pytorch_res, torch_original.Tensor):
            if (
                id(pytorch_res) != id(pytorch_method.__self__)
                or pytorch_res.device.type == pytorch_method.__self__.device.type
            ):
                call_tensor_id.append(id(pytorch_res))
    except Exception as e:
        if align_exception:
            try:
                oneflow_res = oneflow_method(*oneflow_args, **oneflow_kwargs)
            except Exception as ee:
                raise BothDoNotSupportError(e, ee) from None
            print(
                "PyTorch has an error but OneFlow is ok, maybe you should check your implementation to align with PyTorch."
            )
        raise PyTorchDoesNotSupportError(e)
    oneflow_res = oneflow_tensor_eager_run_with_graph_check(
        oneflow, oneflow_method, oneflow_args, oneflow_kwargs, testing_graph, verbose,
    )
    return pytorch_res, oneflow_res


profiled_method_name = []


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
    verbose = os.getenv("ONEFLOW_TEST_VERBOSE") is not None
    pytorch_methods = dir(pytorch)
    if hasattr(pytorch, "__call__") and "__call__" not in pytorch_methods:
        pytorch_methods.append("__call__")
    magic_methods_for_new_cls = {}
    for method_name in pytorch_methods:
        if method_name.startswith("__") and method_name not in skipped_magic_methods:

            def get_dual_method(method_name):
                if method_name == "__call__":

                    if name in profiled_method_name:

                        def method(self, *args, **kwargs):
                            return auto_profiler.profile_dual_object(self)(
                                *args, **kwargs
                            )

                        return method

                    def dual_method(self, *args, **kwargs):
                        param_str = to_string(*args, **kwargs)
                        (
                            pytorch_args,
                            pytorch_kwargs,
                            oneflow_args,
                            oneflow_kwargs,
                        ) = get_args(pytorch, *args, **kwargs)

                        pytorch_res, oneflow_res = get_pytorch_oneflow_res(
                            pytorch,
                            oneflow,
                            pytorch_args,
                            pytorch_kwargs,
                            oneflow_args,
                            oneflow_kwargs,
                            name,
                            verbose,
                            testing_graph,
                            *args,
                        )
                        return GetDualObject(
                            f"{name}({param_str})", pytorch_res, oneflow_res
                        )

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
                        pytorch_res, oneflow_res = get_pytorch_oneflow_tensor_res(
                            pytorch_method,
                            oneflow_method,
                            oneflow,
                            pytorch_args,
                            pytorch_kwargs,
                            oneflow_args,
                            oneflow_kwargs,
                            testing_graph,
                            verbose,
                        )
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


def print_note_fake_program(detail=False):
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
    if detail:
        print(
            f"\033[32m-----------------------------------------------------------\033[0m"
        )
        unique_vis_tensor = []
        flag_vis_input_tensor = [False for _ in range(len(vis_tensor))]
        for i in range(len(vis_tensor)):
            if flag_vis_input_tensor[i] == True:
                continue
            unique_vis_tensor.append(vis_tensor[i])
            flag_vis_input_tensor[i] = True
            for j in range(i + 1, len(vis_tensor)):
                if (
                    id(vis_tensor[i]) == id(vis_tensor[j])
                    and flag_vis_input_tensor[j] == False
                ):
                    flag_vis_input_tensor[j] = True
        unique_extra_tensor = []
        flag_vis_extra_tensor = [False for _ in range(len(extra_input_tensor))]
        for i in range(len(extra_input_tensor)):
            if flag_vis_extra_tensor[i] == True:
                continue
            unique_extra_tensor.append(extra_input_tensor[i])
            flag_vis_extra_tensor[i] = True
            for j in range(i + 1, len(extra_input_tensor)):
                if (
                    id(extra_input_tensor[i]) == id(extra_input_tensor[j])
                    and flag_vis_extra_tensor[j] == False
                ):
                    flag_vis_extra_tensor[j] = True

        print(
            f"\033[32mThis program has {len(unique_extra_tensor) + len(unique_vis_tensor)} input tensor: \033[0m"
        )
        for input_tensor in iter(unique_extra_tensor):
            print(f"\033[32mShape{get_tensor_shape(input_tensor)}\033[0m")
            print(f"\033[32m{input_tensor}\033[0m")
            print(
                f"\033[32m-----------------------------------------------------------\033[0m"
            )
        for input_tensor in iter(unique_vis_tensor):
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
    vis_parameters.clear()
    extra_input_tensor.clear()
    flow.set_printoptions(profile="full")


tensor_size_limit_mb = int(os.getenv("ONEFLOW_TEST_TENSOR_SIZE_LIMIT_MB", 32))


class DualObject:
    def __init__(self, name, pytorch, oneflow):
        self.name = name
        if isinstance(pytorch, torch_original.nn.Module):
            if is_global():
                pytorch.load_state_dict(broadcast(pytorch).state_dict())
            state_dict = pytorch.state_dict()
            state_dict = {k: v.detach().cpu().numpy() for (k, v) in state_dict.items()}
            oneflow_state_dict = oneflow.state_dict()
            oneflow_state_dict = {
                k: v.detach() for (k, v) in oneflow_state_dict.items()
            }
            already_global = any([v.is_global for v in oneflow_state_dict.values()])
            if is_global() and already_global:
                for k, v in state_dict.items():
                    if k not in oneflow_state_dict:
                        continue
                    of_state = oneflow_state_dict[k]
                    if of_state.is_global:
                        state_dict[k] = flow.tensor(
                            v, sbp=of_state.sbp, placement=of_state.placement
                        )

            oneflow.load_state_dict(state_dict, strict=False)

            if is_global():
                if already_global:
                    for (k, v) in oneflow_state_dict.items():
                        if v.is_global:
                            t = getattr(oneflow, k)
                            new = t.to_global(placement=v.placement, sbp=v.sbp)
                            if isinstance(t, flow.nn.Parameter):
                                new = flow.nn.Parameter(new)
                            setattr(
                                oneflow, k, new,
                            )
                else:
                    oneflow = oneflow.to_global(
                        placement=flow.placement.all("cpu"), sbp=[flow.sbp.broadcast,],
                    )
            if testing:
                dual_modules_to_test.append(self)
        if isinstance(pytorch, torch_original.Tensor):
            tensor_size_mb = pytorch.nelement() * pytorch.element_size() / 1024 / 1024
            assert (
                tensor_size_mb < tensor_size_limit_mb
            ), f"Tensor memory in autotest cannot be larger than {tensor_size_limit_mb}MB, but got {tensor_size_mb}MB"
            if testing:
                dual_objects_to_test.append(self)
        self.pytorch = pytorch
        self.oneflow = oneflow

    def __repr__(self):
        return f"PyTorch object:\n{self.pytorch}\n\nOneFlow object:\n{self.oneflow}"

    def __getattr__(self, key):
        if key in ["to_global", "to_local"]:

            def identity(*args, **kwargs):
                if isinstance(self.pytorch, torch_original.Tensor):
                    return self.pytorch.clone()
                return self.pytorch

            pytorch_attr = identity
        elif key in ["placement", "sbp"]:
            pytorch_attr = "unused"
        elif key in ["broadcast_like"]:

            def broadcast_like(x, y, *args, **kwargs):
                return self.pytorch.broadcast_to(x, y.size())

            pytorch_attr = broadcast_like
        else:
            pytorch_attr = getattr(self.pytorch, key)
        oneflow_attr = getattr(self.oneflow, key)
        if pytorch_attr is None:
            assert (
                oneflow_attr is None
            ), f"pytorch value is None for attr {key}, but oneflow is not."
            return None
        if self.name == "":
            new_name = key
        else:
            new_name = f"{self.name}.{key}"
        global call_pytorch
        call_pytorch = self.pytorch
        return GetDualObject(new_name, pytorch_attr, oneflow_attr)

    def __setattr__(self, key, value):
        if isinstance(value, DualObject):
            setattr(self.pytorch, key, value.pytorch)
            setattr(self.oneflow, key, value.oneflow)
        else:
            self.__dict__[key] = value

    def __eq__(self, other):
        if isinstance(other, DualObject):
            return self.pytorch == other.pytorch and self.oneflow == other.oneflow
        else:
            return self.pytorch == other


dual_modules_to_test = []
dual_objects_to_test = []
torch_type2checker = {}


def equality_checker(torch_type, flow_type):
    def deco(f):
        torch_type2checker[torch_type, flow_type] = f
        return f

    return deco


def check_equality(dual_object: DualObject, rtol=0.0001, atol=1e-05, check_dtype=False):
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
    return checker(dual_object.pytorch, dual_object.oneflow, rtol, atol, check_dtype)


@equality_checker(torch_original.Tensor, flow.Tensor)
@equality_checker(torch_original.Tensor, flow._oneflow_internal.Tensor)
def check_tensor_equality(
    torch_tensor, flow_tensor, rtol=0.0001, atol=1e-05, check_dtype=False
):
    if torch_tensor.grad is not None:
        if flow_tensor.grad is None:
            print_note_fake_program(detail=True)
        assert (
            flow_tensor.grad is not None
        ), f"OneFlow tensor doesn't have grad while PyTorch tensor has one, PyTorch tensor is\n {torch_tensor}\n, OneFlow tensor is\n{flow_tensor} "
        torch_grad = torch_tensor.grad.detach().cpu().numpy()
        flow_grad = flow_tensor.grad.numpy()
        if not np.allclose(
            torch_grad, flow_grad, rtol=rtol, atol=atol, equal_nan=True,
        ):
            print_note_fake_program(detail=True)
            print("---------Grad Shape--------")
            print(torch_grad.shape)
            print(flow_grad.shape)
            print(
                f"Grads are not equal. PyTorch grad: \n{torch_grad}\n, OneFlow grad: \n{flow_grad}"
            )
            return False
    torch_numpy = torch_tensor.detach().cpu().numpy()
    oneflow_numpy = flow_tensor.numpy()
    equality_res = np.allclose(
        torch_numpy, oneflow_numpy, rtol=rtol, atol=atol, equal_nan=True,
    )
    # NOTE: if check_dtype=True, then check the equality of data type
    if check_dtype:
        equality_res = equality_res and (torch_numpy.dtype == oneflow_numpy.dtype)

    if equality_res == False:
        print_note_fake_program(detail=True)
        print("---------Tensor Shape--------")
        print(torch_tensor.shape)
        print(flow_tensor.shape)
        print("---------Tensor dtype--------")
        print(torch_tensor.dtype)
        print(flow_tensor.dtype)
    return equality_res


@equality_checker(int, int)
@equality_checker(bool, bool)
def check_basetype_equality(a, b, ignored1, ignored2, check_dtype=False):
    if check_dtype:
        return (a == b) and (type(a) == type(b))
    return a == b


@equality_checker(tuple, tuple)
@equality_checker(list, list)
def check_basetype_equality(a, b, rtol=0.0001, atol=1e-05, check_dtype=False):
    if len(a) != len(b):
        equality_res = False
    else:
        for i in range(len(a)):
            torch_np = a[i].detach().cpu().numpy()
            flow_np = b[i].detach().cpu().numpy()
            equality_res = np.allclose(
                torch_np, flow_np, rtol=rtol, atol=atol, equal_nan=True,
            )
            if check_dtype:
                equality_res = equality_res and (torch_np.dtype == flow_np.dtype)
            if equality_res == False:
                print_note_fake_program(detail=True)
                print("---------Tensor Shape--------")
                print(a[i].shape)
                print(b[i].shape)
                print("---------Tensor dtype--------")
                print(a[i].dtype)
                print(b[i].dtype)
                break

    return equality_res


@equality_checker(type(None), type(None))
def check_nonetype_equality(a, b, ignored1, ignored2, check_dtype=False):
    return True


def autotest(
    n=20,
    auto_backward: Union[bool, str] = True,
    rtol=0.0001,
    atol=1e-05,
    check_graph=True,
    check_allclose=True,
    check_dtype=False,
    check_grad_use_random_data=True,
):
    verbose = os.getenv("ONEFLOW_TEST_VERBOSE") is not None

    if check_graph == "ValidatedFalse":
        # check graph is intentionally closed and there is a validated reason.
        check_graph = False

    def deco(f):
        @functools.wraps(f)
        def new_f(test_case, *args, **kwargs):
            successful_runs_needed = n
            loop_limit = successful_runs_needed * 20
            current_run = 0
            while successful_runs_needed > 0:
                clear_note_fake_program()
                if current_run > loop_limit:
                    raise ValueError(
                        "autotest stuck in an endless loop, usually it is caused by invalid code in the test case"
                    )
                dual_modules_to_test.clear()
                dual_objects_to_test.clear()
                global global_check_allclose, global_rtol, global_atol, global_backward
                global_check_allclose = check_allclose
                global_rtol = rtol
                global_atol = atol
                global_backward = auto_backward

                try:
                    global testing_graph
                    # for generate fake program input tensor
                    global testing
                    testing = True
                    if check_graph:
                        testing_graph = True
                    res = f(test_case, *args, **kwargs)
                    testing = False
                    testing_graph = False
                except (PyTorchDoesNotSupportError, BothDoNotSupportError) as e:
                    if verbose:
                        print(f"{f.__name__}")
                        print(e)
                    current_run += 1
                    continue
                if res is not None:
                    if not isinstance(res, collections.abc.Sequence):
                        res = [res]
                    for x in res:
                        if x is None:
                            continue
                        if auto_backward:
                            if isinstance(x.pytorch, torch_original.Tensor):
                                if auto_backward == "auto" and (
                                    not x.pytorch.requires_grad
                                    or not x.oneflow.requires_grad
                                ):
                                    continue
                                call_tensor_id.append(id(x.pytorch))
                                if check_grad_use_random_data:
                                    np_arr = rng.uniform(
                                        low=0, high=1, size=list(x.oneflow.shape)
                                    )
                                    if is_global():
                                        np_arr = broadcast(np_arr)
                                        flow_tensor = flow.tensor(
                                            np_arr,
                                            dtype=x.oneflow.dtype,
                                            placement=x.oneflow.placement,
                                            sbp=len(x.oneflow.sbp)
                                            * [flow.sbp.broadcast],
                                        )
                                    else:
                                        flow_tensor = flow.tensor(
                                            np_arr,
                                            dtype=x.oneflow.dtype,
                                            device=x.oneflow.device,
                                        )
                                    # TODO(): Inferred shape of some op is different between oneflow and torch
                                    pytorch_tensor = torch_original.tensor(
                                        np_arr.reshape(list(x.pytorch.shape)),
                                        dtype=x.pytorch.dtype,
                                        device=x.pytorch.device,
                                    )
                                    call_tensor_id.append(id(pytorch_tensor))
                                    diff_output = GetDualObject(
                                        "unused", pytorch_tensor, flow_tensor
                                    )
                                    x.backward(diff_output)
                                else:
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
                        test_case.assertTrue(
                            check_equality(
                                x, rtol=rtol, atol=atol, check_dtype=check_dtype,
                            ),
                            x,
                        )

                if verbose:
                    print(f"{f.__name__} test eager passed.")

                if verbose and check_graph:
                    print(f"{f.__name__} test graph passed.")

                successful_runs_needed -= 1
                current_run += 1

        return new_f

    return deco


def globaltest(f):
    @functools.wraps(f)
    def new_f(*args, **kwargs):
        with GlobalScope() as scope:
            return f(*args, **kwargs)

    return new_f


def random_tensor(
    ndim=None,
    dim0=1,
    dim1=None,
    dim2=None,
    dim3=None,
    dim4=None,
    low=None,
    high=None,
    dtype=float,
    requires_grad=True,
    pin_memory=False,
):
    if isinstance(requires_grad, generator):
        requires_grad = requires_grad.value()
    pytorch_tensor = (
        random_pytorch_tensor(
            ndim, dim0, dim1, dim2, dim3, dim4, low, high, dtype, pin_memory
        )
        .value()
        .requires_grad_(requires_grad and dtype != int)
    )
    extra_input_tensor.append(pytorch_tensor)
    if is_global():
        flow_tensor = flow.tensor(
            pytorch_tensor.detach().cpu().numpy(),
            requires_grad=(requires_grad and dtype != int),
            placement=flow.placement.all("cpu"),
            sbp=flow.sbp.broadcast,
        )
    else:
        flow_tensor = flow.tensor(
            pytorch_tensor.detach().cpu().numpy(),
            requires_grad=(requires_grad and dtype != int),
            pin_memory=pin_memory,
        )

    return GetDualObject("unused", pytorch_tensor, flow_tensor)


def random_dtype(seq_names):
    pytorch_dtype = random_pytorch_dtype(seq_names).value()
    if pytorch_dtype is None:
        flow_dtype = None
    else:
        flow_dtype = type_name_to_flow_type[pytorch_dtype.__str__().split(".")[-1]]
    return GetDualObject("DualDType", pytorch_dtype, flow_dtype)


def choice_tensor(
    a, size=None, replace=True, p=None, dtype=int, requires_grad=False,
):
    """Generates a random sample from a given 1-D array, which aligns with numpy.random.choice
    see https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html for details

    """
    if isinstance(requires_grad, generator):
        requires_grad = requires_grad.value()
    pytorch_tensor = (
        choice_pytorch_tensor(a, size, replace, p, dtype)
        .value()
        .requires_grad_(requires_grad and dtype != int)
    )
    if is_global():
        flow_tensor = flow.tensor(
            pytorch_tensor.detach().cpu().numpy(),
            requires_grad=(requires_grad and dtype != int),
            placement=flow.placement.all("cpu"),
            sbp=flow.sbp.broadcast,
        )
    else:
        flow_tensor = flow.tensor(
            pytorch_tensor.detach().cpu().numpy(),
            requires_grad=(requires_grad and dtype != int),
        )

    return GetDualObject("unused", pytorch_tensor, flow_tensor)


torch = GetDualObject("", torch_original, flow)
__all__ = ["autotest", "globaltest", "random_tensor", "random_dtype", "choice_tensor"]
