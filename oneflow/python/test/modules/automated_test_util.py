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
import typing
from typing import Dict, List, Optional, Tuple, Any, Union
import random as random_util

import oneflow.experimental as flow

flow.enable_eager_execution()
import torch
import numpy as np


from numpy.random import default_rng

rng = default_rng()

default_generators = {}


def data_generator(annotation):
    def register_data_generator(func):
        default_generators[annotation] = func
        return func

    return register_data_generator


@data_generator(bool)
def generate_bool():
    val = choose([True, False])(None)
    return val, val


def random_4d_tensor(batch_size=1, channels=4, height=5, width=6):
    def generator(_):
        np_arr = rng.random((batch_size, channels, height, width))
        return flow.Tensor(np_arr), torch.Tensor(np_arr)

    return generator


def choose(x):
    def generator(_):
        return random_util.choice(x)

    return generator


def random(low, high):
    def generator(annotation):
        if hasattr(annotation, "__origin__"):
            annotation = eval(repr(annotation))
            if annotation.__origin__ is Union:
                x = choose(annotation.__args__)(None)
                return generator(x)
            if annotation.__origin__ is Tuple:
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
    func_name,
    extra_annotations: Optional[Dict[str, Any]] = None,
    extra_generators: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    training: bool = True,
    rtol=1e-5,
    atol=1e-8,
    n=1,
):
    assert device in ["cuda", "cpu"]
    if extra_annotations is None:
        extra_annotations = {}
    if extra_generators is None:
        extra_generators = {}

    torch_func = eval(f"torch.{func_name}")
    spec = inspect.getfullargspec(torch_func)
    annotations = spec.annotations
    annotations.update(extra_annotations)
    if "return" in annotations:
        del annotations["return"]
    args = (set(spec.args) | set(spec.kwonlyargs)) - {"self"}
    assert args == set(
        annotations.keys()
    ), f"args = {args}, annotations = {annotations.keys()}"
    annotations.update({"input": torch.Tensor})

    def generate(name):
        annotation = annotations[name]
        if name in extra_generators:
            return extra_generators[name](annotation)
        return default_generators[annotation]()

    while n > 0:
        flow_param_dict = {}
        torch_param_dict = {}
        for name in args:
            flow_data, torch_data = generate(name)
            flow_param_dict[name] = flow_data
            torch_param_dict[name] = torch_data

        flow_input, torch_input = generate("input")
        flow_input, torch_input = flow_input.to(device), torch_input.to(device)
        try:
            torch_module = torch_func(**torch_param_dict)
            torch_module = torch_module.to(device)
            torch_module.train(training)
            torch_res = torch_module(torch_input)
            state_dict = torch_module.state_dict()
            state_dict = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
        except:
            continue

        flow_func = eval(f"flow.{func_name}")
        flow_module = flow_func(**flow_param_dict)
        flow_module = flow_module.to(device)
        flow_module.train(training)
        flow_module.load_state_dict(state_dict)
        flow_res = flow_module(flow_input)
        is_allclose = np.allclose(
            flow_res.numpy(), torch_res.detach().cpu().numpy(), rtol=rtol, atol=atol
        )
        n -= 1
        if not is_allclose:
            return False, (torch_input, torch_param_dict, state_dict)
    return True, None
