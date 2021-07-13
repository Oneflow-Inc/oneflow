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
import inspect
import typing  # This unused import is needed
from typing import Dict, Optional, Tuple, Any, Union
from collections import namedtuple
import random as random_util
import os

import oneflow.experimental as flow
import torch
import numpy as np

TEST_MODULE = 0
TEST_FLOW = 1
TEST_TENSOR = 2

rng = np.random.default_rng()

default_generators = {}


def data_generator(annotation):
    def register_data_generator(func):
        default_generators[annotation] = func
        return func

    return register_data_generator


@data_generator(bool)
def _random_bool():
    val = random_util.choice([True, False])
    return val, val


@data_generator(torch.Tensor)
def _random_tensor():
    return random_tensor()(None)


def random_tensor(ndim=None, dim0=1, dim1=None, dim2=None, dim3=None, dim4=None):
    assert ndim is None or 1 <= ndim <= 5
    if ndim is None:
        ndim = rng.integers(low=1, high=6)
    shape = rng.integers(low=1, high=8, size=ndim)
    if dim0 is not None:
        shape[0] = dim0
    if ndim >= 2 and dim1 is not None:
        shape[1] = dim1
    if ndim >= 3 and dim2 is not None:
        shape[2] = dim2
    if ndim >= 4 and dim3 is not None:
        shape[3] = dim3
    if ndim == 5 and dim4 is not None:
        shape[4] = dim4

    def generator(_):
        np_arr = rng.random(shape)
        return flow.Tensor(np_arr), torch.Tensor(np_arr)

    return generator


def choose(x):
    def generator(_):
        val = random_util.choice(x)
        return val, val

    return generator


def random(low, high):
    def generator(annotation):
        if hasattr(annotation, "__origin__"):
            # PyTorch _size_2_t and similar types are defined by type variables,
            # leading to unexpected __args__ and __origin__
            #
            # _size_2_t = Union[T, Tuple[T, T]][int]
            # _size_2_t.__origin__
            # >> typing.Union[~T, typing.Tuple[~T, ~T]]
            #
            # So recreate a new annotation object by repr and eval
            #
            # _size_2_t
            # >> typing.Union[int, typing.Tuple[int, int]]
            # _size_2_t_new = eval(repr(annotation))
            # _size_2_t_new.__origin__
            # >> typing.Union
            annotation = eval(repr(annotation))
            if annotation.__origin__ is Union:
                x = random_util.choice(annotation.__args__)
                return generator(x)
            if annotation.__origin__ is Tuple or annotation.__origin__ is tuple:
                t = [generator(x) for x in annotation.__args__]
                return zip(*t)
            else:
                raise NotImplementedError(
                    f"Not implemented annotation {annotation} in random, type(annotation.__origin__) is {type(annotation.__origin__)}"
                )

        if annotation == int:
            val = int(rng.integers(low, high))
        elif annotation == float:
            val = float(rng.random() * (high - low) + low)
        else:
            raise NotImplementedError(
                f"Not implemented annotation {annotation} in random"
            )
        return val, val

    return generator


def constant(val):
    def generator(_):
        return val, val

    return generator


def test_against_pytorch(
    test_case,
    callable_name,
    extra_annotations: Optional[Dict[str, Any]] = None,
    extra_generators: Optional[Dict[str, Any]] = None,
    extra_defaults: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    training: bool = True,
    backward: bool = True,
    rtol=1e-4,
    atol=1e-5,
    n=20,
    pytorch_callable_name=None,
    api_flag: int = TEST_MODULE,
):
    assert device in ["cuda", "cpu"]
    if not training:
        assert not backward
    if extra_annotations is None:
        extra_annotations = {}
    if extra_generators is None:
        extra_generators = {}
    if extra_defaults is None:
        extra_defaults = {}
    if pytorch_callable_name is None:
        pytorch_callable_name = callable_name

    verbose = os.getenv("ONEFLOW_TEST_VERBOSE") is not None

    def has_full_args_spec(callable):
        try:
            spec = inspect.getfullargspec(callable)
            return True
        except Exception:
            return False

    if api_flag == TEST_TENSOR:
        pytorch_tensor = torch.Tensor(1)
        pytorch_call = eval(f"pytorch_tensor.{pytorch_callable_name}")
    else:
        pytorch_call = eval(f"torch.{pytorch_callable_name}")

    if has_full_args_spec(pytorch_call):
        spec = inspect.getfullargspec(pytorch_call)
    else:
        Spec = namedtuple(
            "spec",
            "args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations",
        )
        args = list(extra_annotations.keys()) + list(extra_defaults.keys())
        spec = Spec(args, None, None, [], [], {}, {})

    annotations = spec.annotations
    annotations.update(extra_annotations)

    if "return" in annotations:
        del annotations["return"]

    args = (set(spec.args) | set(spec.kwonlyargs)) - {"self"}

    assert args == set(
        annotations.keys()
    ), f"args = {args}, annotations = {annotations.keys()}"
    annotations.update({"input": torch.Tensor})

    def has_default(name):
        if name in spec.args:
            return (len(spec.args) - spec.args.index(name)) <= len(spec.defaults)
        else:
            assert name in spec.kwonlyargs
            return (len(spec.kwonlyargs) - spec.kwonlyargs.index(name)) <= len(
                spec.kwonlydefaults
            )

    def generate(name):
        annotation = annotations[name]
        if name in extra_generators:
            return extra_generators[name](annotation)
        return default_generators[annotation]()

    while n > 0:
        flow_attr_dict = {}
        torch_attr_dict = {}
        for name in args:
            if has_default(name):
                if rng.random() < 1 / 3:
                    continue
            flow_data, torch_data = generate(name)
            if isinstance(torch_data, torch.Tensor):
                torch_data = torch_data.to(device)
            if isinstance(flow_data, flow.Tensor):
                flow_data = flow_data.to(device)
            flow_attr_dict[name] = flow_data
            torch_attr_dict[name] = torch_data

        if verbose:
            print(f"attr = {torch_attr_dict}, device = {device}")

        flow_input_original, torch_input_original = generate("input")
        flow_input_original.requires_grad_(backward)
        torch_input_original.requires_grad_(backward)
        flow_input, torch_input = (
            flow_input_original.to(device),
            torch_input_original.to(device),
        )
        try:
            if api_flag == TEST_MODULE:
                torch_call = pytorch_call(**torch_attr_dict)
                torch_call = torch_call.to(device)
                torch_call.train(training)
                torch_res = torch_call(torch_input)
                state_dict = torch_call.state_dict()
                state_dict = {
                    k: v.detach().cpu().numpy() for k, v in state_dict.items()
                }
            elif api_flag == TEST_FLOW:
                torch_xxx_func = eval(f"torch.{pytorch_callable_name}")
                torch_res = torch_xxx_func(torch_input, **torch_attr_dict)

            else:
                torch_tensor_xxx_func = eval(f"torch_input.{pytorch_callable_name}")
                torch_res = torch_tensor_xxx_func(**torch_attr_dict)

            loss = torch_res.sum()
            loss.backward()
            if api_flag == TEST_MODULE:
                state_dict = torch_call.state_dict()
                state_dict = {
                    k: v.detach().cpu().numpy() for k, v in state_dict.items()
                }
        except Exception as e:
            if verbose:
                print(f"PyTorch error: {e}")
            # The random generated test data is not always valid,
            # so just skip when PyTorch raises an exception
            continue

        if api_flag == TEST_MODULE:
            flow_call_class = eval(f"flow.{callable_name}")
            flow_call = flow_call_class(**flow_attr_dict)
            flow_call = flow_call.to(device)
            flow_call.train(training)
            flow_call.load_state_dict(state_dict)
            flow_res = flow_call(flow_input)
        elif api_flag == TEST_FLOW:
            flow_xxx_func = eval(f"flow.{callable_name}")
            flow_res = flow_xxx_func(flow_input, **flow_attr_dict)
        else:
            flow_tensor_xxx_func = eval(f"flow_input.{callable_name}")
            flow_res = flow_tensor_xxx_func(**flow_attr_dict)

        loss = flow_res.sum()
        loss.backward()

        def allclose_or_fail(flow_tensor, torch_tensor):
            is_allclose = np.allclose(
                flow_tensor.numpy(),
                torch_tensor.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
            )
            test_case.assertTrue(
                is_allclose,
                f"flow_tensor = {flow_tensor},\ntorch_tensor = {torch_tensor},\nattr_dict = {torch_attr_dict},\nflow_input_tensor = {flow_input_original}",
            )

        allclose_or_fail(flow_res, torch_res)
        allclose_or_fail(flow_input_original.grad, torch_input_original.grad)
        if api_flag == TEST_MODULE:
            flow_parameters = dict(flow_call.named_parameters())
            for name, torch_param in torch_call.named_parameters():
                flow_param = flow_parameters[name]
                allclose_or_fail(flow_param.grad, torch_param.grad)
        n -= 1


def test_module_against_pytorch(
    test_case,
    callable_name,
    extra_annotations: Optional[Dict[str, Any]] = None,
    extra_generators: Optional[Dict[str, Any]] = None,
    extra_defaults: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    training: bool = True,
    backward: bool = True,
    rtol=1e-4,
    atol=1e-5,
    n=20,
    pytorch_callable_name=None,
):
    return test_against_pytorch(
        test_case=test_case,
        callable_name=callable_name,
        extra_annotations=extra_annotations,
        extra_generators=extra_generators,
        extra_defaults=extra_defaults,
        device=device,
        training=training,
        backward=backward,
        rtol=rtol,
        atol=atol,
        n=n,
        pytorch_callable_name=pytorch_callable_name,
        api_flag=TEST_MODULE,
    )


def test_flow_against_pytorch(
    test_case,
    callable_name,
    extra_annotations: Optional[Dict[str, Any]] = None,
    extra_generators: Optional[Dict[str, Any]] = None,
    extra_defaults: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    training: bool = True,
    backward: bool = True,
    rtol=1e-4,
    atol=1e-5,
    n=20,
    pytorch_callable_name=None,
):
    return test_against_pytorch(
        test_case=test_case,
        callable_name=callable_name,
        extra_annotations=extra_annotations,
        extra_generators=extra_generators,
        extra_defaults=extra_defaults,
        device=device,
        training=training,
        backward=backward,
        rtol=rtol,
        atol=atol,
        n=n,
        pytorch_callable_name=pytorch_callable_name,
        api_flag=TEST_FLOW,
    )


def test_tensor_against_pytorch(
    test_case,
    callable_name,
    extra_annotations: Optional[Dict[str, Any]] = None,
    extra_generators: Optional[Dict[str, Any]] = None,
    extra_defaults: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    training: bool = True,
    backward: bool = True,
    rtol=1e-4,
    atol=1e-5,
    n=20,
    pytorch_callable_name=None,
):
    return test_against_pytorch(
        test_case=test_case,
        callable_name=callable_name,
        extra_annotations=extra_annotations,
        extra_generators=extra_generators,
        extra_defaults=extra_defaults,
        device=device,
        training=training,
        backward=backward,
        rtol=rtol,
        atol=atol,
        n=n,
        pytorch_callable_name=pytorch_callable_name,
        api_flag=TEST_TENSOR,
    )


__all__ = [
    "random_tensor",
    "random",
    "choose",
    "constant",
    "test_module_against_pytorch",
    "test_flow_against_pytorch",
    "test_tensor_against_pytorch",
]
