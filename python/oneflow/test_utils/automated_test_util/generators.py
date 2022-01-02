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
import os
import random as random_util
import typing
from collections import namedtuple
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

import oneflow as flow

py_tuple = tuple
NoneType = type(None)

TEST_MODULE = 0
TEST_FLOW = 1
TEST_TENSOR = 2
rng = np.random.default_rng()
annotation2default_generator = {}
annotation2torch_to_flow_converter = {}
NoneType = type(None)


def data_generator(annotation):
    def register_data_generator(cls):
        annotation2default_generator[annotation] = lambda: cls()
        return cls

    return register_data_generator


def torch_to_flow_converter(annotation):
    def register_flow_to_flow_converter(func):
        annotation2torch_to_flow_converter[annotation] = func
        return func

    return register_flow_to_flow_converter


@torch_to_flow_converter(torch.Tensor)
def tensor_converter(torch_tensor):
    return flow.tensor(torch_tensor.cpu().numpy())


def convert_torch_object_to_flow(x):
    for (annotation, converter) in annotation2torch_to_flow_converter.items():
        if isinstance(x, annotation):
            return converter(x)
    return x


def pack(x):
    if isinstance(x, generator):
        return x
    return constant(x)


class Nothing:
    pass


class generator:
    def __init__(self, children):
        self.children = children
        self._value = None

    def _init(self):
        self._value = None
        for x in self.children:
            x._init()

    def eval(self):
        self._init()
        return self.value()

    def _calc_value(self):
        raise NotImplementedError()

    def value(self):
        if self._value is None:
            self._value = self._calc_value()
        return self._value

    def size(self):
        return 1

    def __or__(self, other):
        other = pack(other)
        return oneof(
            self, other, possibility=self.size() / (self.size() + other.size())
        )

    def __ror__(self, other):
        return self | other

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + neg(other)

    def __rsub__(self, other):
        return neg(self - other)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return self * other

    def to(self, annotation):
        self._to(annotation)
        for x in self.children:
            x.to(annotation)
        return self

    def _to(self, annotation):
        pass


class add(generator):
    def __init__(self, a, b):
        self.a = pack(a)
        self.b = pack(b)
        super().__init__([self.a, self.b])

    def _calc_value(self):
        return self.a.value() + self.b.value()


class mul(generator):
    def __init__(self, a, b):
        self.a = pack(a)
        self.b = pack(b)
        super(mul, self).__init__([self.a, self.b])

    def _calc_value(self):
        return self.a.value() * self.b.value()


class neg(generator):
    def __init__(self, a):
        self.a = pack(a)
        super().__init__([self.a])

    def _calc_value(self):
        return -self.a.value()


class oneof(generator):
    def __init__(self, *args, possibility=None):
        self.args = list(map(pack, args))
        super().__init__(self.args)
        if isinstance(possibility, float):
            assert len(args) == 2
            possibility = [possibility, 1 - possibility]
        if possibility is None:
            possibility = [1 / len(args)] * len(args)
        self.possibility = pack(possibility)

    def _calc_value(self):
        rand = rng.random()
        sum = 0
        for (i, possibility) in enumerate(self.possibility.value()):
            sum += possibility
            if sum > rand:
                return self.args[i].value()
        raise RuntimeError()

    def size(self):
        return sum([x.size() for x in self.args])


class tuple(generator):
    def __init__(self, *args):
        self.args = list(map(pack, args))
        super().__init__(self.args)

    def _calc_value(self):
        return py_tuple([x.value() for x in self.args])


class constant(generator):
    def __init__(self, x):
        super().__init__([])
        self.x = x

    def _calc_value(self):
        return self.x


class nothing(generator):
    def __init__(self):
        super().__init__([])

    def _calc_value(self):
        return Nothing()


class random(generator):
    def __init__(self, low=1, high=6):
        self.low = pack(low)
        self.high = pack(high)
        super().__init__([self.low, self.high])
        self.annotation = None

    def _to(self, annotation):
        if self.annotation is not None:
            return
        if hasattr(annotation, "__origin__"):
            annotation = eval(repr(annotation))
        self.annotation = annotation

    def _generate(self, annotation):
        if hasattr(annotation, "__origin__"):
            if annotation.__origin__ is Union:
                x = random_util.choice(annotation.__args__)
                return self._generate(x)
            if annotation.__origin__ is Tuple or annotation.__origin__ is py_tuple:
                return [self._generate(x) for x in annotation.__args__]
            else:
                raise NotImplementedError(
                    f"Not implemented annotation {annotation} in random, type(annotation.__origin__) is {type(annotation.__origin__)}"
                )
        (low, high) = (self.low.value(), self.high.value())
        if annotation == int:
            val = int(rng.integers(low, high))
        elif annotation == float:
            val = float(rng.random() * (high - low) + low)
        elif annotation == bool:
            val = random_util.choice([True, False])
        elif annotation is None:
            val = None
        elif annotation is NoneType:
            val = None
        else:
            raise NotImplementedError(
                f"Not implemented annotation {annotation} in random"
            )
        return val

    def _calc_value(self):
        return self._generate(self.annotation)


def random_or_nothing(low, high):
    return oneof(random(low, high), nothing(), possibility=2 / 3)


@data_generator(torch.Tensor)
class random_tensor(generator):
    def __init__(
        self,
        ndim=None,
        dim0=1,
        dim1=None,
        dim2=None,
        dim3=None,
        dim4=None,
        low=0,
        high=1,
        dtype=float,
    ):
        if ndim is None:
            ndim = random(1, 6)
        if dim0 is None:
            dim0 = random(1, 8)
        if dim1 is None:
            dim1 = random(1, 8)
        if dim2 is None:
            dim2 = random(1, 8)
        if dim3 is None:
            dim3 = random(1, 8)
        if dim4 is None:
            dim4 = random(1, 8)
        self.ndim = pack(ndim).to(int)
        self.dim0 = pack(dim0).to(int)
        self.dim1 = pack(dim1).to(int)
        self.dim2 = pack(dim2).to(int)
        self.dim3 = pack(dim3).to(int)
        self.dim4 = pack(dim4).to(int)
        self.low = pack(low).to(float)
        self.high = pack(high).to(float)
        self.dtype = pack(dtype)
        super().__init__(
            [
                self.ndim,
                self.dim0,
                self.dim1,
                self.dim2,
                self.dim3,
                self.dim4,
                self.low,
                self.high,
                self.dtype,
            ]
        )

    def _calc_value(self):
        ndim = self.ndim.value()
        dim0 = self.dim0.value()
        dim1 = self.dim1.value()
        dim2 = self.dim2.value()
        dim3 = self.dim3.value()
        dim4 = self.dim4.value()
        low = self.low.value()
        high = self.high.value()
        dtype = self.dtype.value()
        shape = rng.integers(low=1, high=8, size=ndim)
        if dim0 is not None:
            shape[0] = dim0
        if ndim >= 2:
            shape[1] = dim1
        if ndim >= 3:
            shape[2] = dim2
        if ndim >= 4:
            shape[3] = dim3
        if ndim == 5:
            shape[4] = dim4
        if dtype == float:
            np_arr = rng.uniform(low=low, high=high, size=shape)
            return torch.Tensor(np_arr)
        elif dtype == int:
            np_arr = rng.integers(low=low, high=high, size=shape)
            return torch.tensor(np_arr, dtype=torch.int64)
        else:
            raise NotImplementedError(f"Not implemented dtype {dtype} in random")


@data_generator(bool)
def random_bool():
    return random().to(bool)


class random_device(generator):
    def __init__(self):
        super().__init__([])

    def _calc_value(self):
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            return "cpu"
        else:
            return random_util.choice(["cuda", "cpu"])


class cpu_device(generator):
    def __init__(self):
        super().__init__([])

    def _calc_value(self):
        return random_util.choice(["cpu"])


class gpu_device(generator):
    def __init__(self):
        super().__init__([])

    def _calc_value(self):
        return random_util.choice(["cuda"])


def test_against_pytorch(
    test_case,
    callable_name,
    extra_annotations: Optional[Dict[str, Any]] = None,
    extra_generators: Optional[Dict[str, Any]] = None,
    extra_defaults: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    training: bool = True,
    backward: bool = True,
    rtol=0.0001,
    atol=1e-05,
    n=20,
    pytorch_callable_name=None,
    api_flag: int = TEST_MODULE,
):
    assert device in ["cuda", "cpu"]
    if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
        device = "cpu"
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
            inspect.getfullargspec(callable)
            return True
        except Exception:
            return False

    if api_flag == TEST_TENSOR:
        pytorch_tensor = torch.Tensor(1)
        pytorch_call = eval(f"pytorch_tensor.{pytorch_callable_name}")
    else:
        pytorch_call = eval(f"torch.{pytorch_callable_name}")
    Spec = namedtuple(
        "spec",
        "args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations",
    )
    if has_full_args_spec(pytorch_call):
        tmp_spec = inspect.getfullargspec(pytorch_call)
        new_defaults = tmp_spec.defaults
        if new_defaults is None:
            new_defaults = []
        new_kwonlydefaults = tmp_spec.kwonlydefaults
        if new_kwonlydefaults is None:
            new_kwonlydefaults = []
        spec = Spec(
            tmp_spec.args,
            tmp_spec.varargs,
            tmp_spec.varkw,
            new_defaults,
            tmp_spec.kwonlyargs,
            new_kwonlydefaults,
            tmp_spec.annotations,
        )
    else:
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
    if "input" not in annotations:
        annotations.update({"input": torch.Tensor})

    def has_default(name):
        if name in spec.args:
            return len(spec.args) - spec.args.index(name) <= len(spec.defaults)
        else:
            assert name in spec.kwonlyargs
            return len(spec.kwonlyargs) - spec.kwonlyargs.index(name) <= len(
                spec.kwonlydefaults
            )

    def get_generator(name):
        annotation = annotations[name]
        if name in extra_generators:
            generator = extra_generators[name]
        else:
            generator = annotation2default_generator[annotation]()
        generator = generator.to(annotation)
        return generator

    while n > 0:
        flow_attr_dict = {}
        torch_attr_dict = {}
        generator_tuple = tuple(
            *[get_generator(name) for name in args] + [get_generator("input")]
        )
        values = generator_tuple.eval()
        for (i, name) in enumerate(args):
            torch_data = values[i]
            if isinstance(torch_data, Nothing):
                continue
            flow_data = convert_torch_object_to_flow(torch_data)
            if isinstance(torch_data, torch.Tensor):
                torch_data = torch_data.to(device)
            if isinstance(flow_data, flow.Tensor):
                flow_data = flow_data.to(device)
            flow_attr_dict[name] = flow_data
            torch_attr_dict[name] = torch_data
        if verbose:
            print(f"attr = {torch_attr_dict}, device = {device}")
        torch_input_original = values[-1]
        flow_input_original = convert_torch_object_to_flow(torch_input_original)
        flow_input_original.requires_grad_(backward)
        torch_input_original.requires_grad_(backward)
        (flow_input, torch_input) = (
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
                    k: v.detach().cpu().numpy() for (k, v) in state_dict.items()
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
                    k: v.detach().cpu().numpy() for (k, v) in state_dict.items()
                }
        except Exception as e:
            if verbose:
                print(f"PyTorch error: {e}")
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
            for (name, torch_param) in torch_call.named_parameters():
                flow_param = flow_parameters[name]
                allclose_or_fail(flow_param.grad, torch_param.grad)
        if verbose:
            print("test passed")
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
    rtol=0.0001,
    atol=1e-05,
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
    rtol=0.0001,
    atol=1e-05,
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
    rtol=0.0001,
    atol=1e-05,
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
    "random_bool",
    "random_device",
    "cpu_device",
    "gpu_device",
    "random",
    "random_or_nothing",
    "oneof",
    "constant",
    "nothing",
    "test_module_against_pytorch",
    "test_flow_against_pytorch",
    "test_tensor_against_pytorch",
]
