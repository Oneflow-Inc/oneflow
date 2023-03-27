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
from typing import Any, Dict, Optional, Tuple, Sequence, Union
from itertools import product

import numpy as np
import torch

import oneflow as flow

from .global_scope import *
from .util import broadcast

py_tuple = tuple
NoneType = type(None)

TEST_MODULE = 0
TEST_FLOW = 1
TEST_TENSOR = 2
rng = np.random.default_rng()
annotation2default_generator = {}
annotation2torch_to_flow_converter = {}
NoneType = type(None)
random_value_default_range = {int: (-10, 11), float: (-1, 1)}


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
        self._has_value = False

    def _init(self):
        self._value = None
        self._has_value = False
        for x in self.children:
            x._init()

    def eval(self):
        self._init()
        return self.value()

    def _calc_value(self):
        raise NotImplementedError()

    def value(self):
        if not self._has_value:
            self._value = self._calc_value()
            if is_global():
                self._value = broadcast(self._value)
            self._has_value = True
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

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._calc_value()(*args, **kwds)

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
class random_pytorch_tensor(generator):
    def __init__(
        self,
        ndim=None,
        dim0=1,
        dim1=None,
        dim2=None,
        dim3=None,
        dim4=None,
        low=None,
        high=None,
        dtype=float,
        pin_memory=False,
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
        self.pin_memory = pin_memory
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
                self.pin_memory,
            ]
        )

    def _calc_value(self):
        ndim = self.ndim.value()
        dim0 = self.dim0.value()
        dim1 = self.dim1.value()
        dim2 = self.dim2.value()
        dim3 = self.dim3.value()
        dim4 = self.dim4.value()
        dtype = self.dtype.value()
        low = self.low.value()
        high = self.high.value()
        if low is None:
            low = random_value_default_range[dtype][0]
        if high is None:
            high = random_value_default_range[dtype][1]
        pin_memory = self.pin_memory

        shape = rng.integers(low=1, high=8, size=ndim)
        if ndim == 0:
            shape = []
        if ndim >= 1 and dim0 is not None:
            shape[0] = dim0
        if ndim >= 2:
            shape[1] = dim1
        if ndim >= 3:
            shape[2] = dim2
        if ndim >= 4:
            shape[3] = dim3
        if ndim == 5:
            shape[4] = dim4

        pytorch_tensor = None
        if dtype == float:
            np_arr = rng.uniform(low=low, high=high, size=shape)
            res = torch.Tensor(np_arr)
            if pin_memory:
                res = res.pin_memory()
            return res
        elif dtype == int:
            np_arr = rng.integers(low=low, high=high, size=shape)
            res = torch.tensor(np_arr, dtype=torch.int64)
            if pin_memory:
                res = res.pin_memory()
            return res
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


@data_generator(torch.dtype)
class random_pytorch_dtype(generator):
    none_dtype_seq = [None]
    bool_dtype_seq = [torch.bool]
    floating_dtype_seq = [torch.float, torch.double]
    half_dtype_seq = [torch.half]
    bfloat16_dtype_seq = [torch.bfloat16]
    signed_int_dtype_seq = [torch.int8, torch.int32, torch.int64]
    unsigned_int_dtype_seq = [torch.uint8]
    int_dtype_seq = [torch.int8, torch.int32, torch.int64]
    image_dtype_seq = [torch.uint8, torch.float]
    index_dtype_seq = [torch.int32, torch.int64]
    arithmetic_dtype_seq = [*floating_dtype_seq, *int_dtype_seq]
    pod_dtype_seq = [*arithmetic_dtype_seq, *unsigned_int_dtype_seq, *bool_dtype_seq]
    all_dtype_seq = [*arithmetic_dtype_seq, torch.half, torch.bfloat16]

    seq_name_to_seq = {
        "None": none_dtype_seq,
        "bool": bool_dtype_seq,
        "float": floating_dtype_seq,
        "half": half_dtype_seq,
        "bfloat16": bfloat16_dtype_seq,
        "signed": signed_int_dtype_seq,
        "unsigned": unsigned_int_dtype_seq,
        "int": int_dtype_seq,
        "image": image_dtype_seq,
        "index": index_dtype_seq,
        "arithmetic": arithmetic_dtype_seq,
        "pod": pod_dtype_seq,
        "all": all_dtype_seq,
    }

    def __init__(self, seq_names):
        super().__init__([])
        # concat related dtype_seq for name in seq_names
        self.data_type_seq = [
            dtype for name in seq_names for dtype in self.seq_name_to_seq[name]
        ]

    def _calc_value(self):
        return random_util.choice(self.data_type_seq)


class all_placement(generator):
    def __init__(self):
        super().__init__([])
        self.node_size = flow.env.get_node_size()
        self.world_size = flow.env.get_world_size()
        self.num_rank_for_each_node = self.world_size // self.node_size

    def __len__(self):
        return len(self.value())

    def __getitem__(self, key):
        return self.value()[key]

    def _calc_device(self):
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            return [
                "cpu",
            ]
        else:
            return ["cuda", "cpu"]

    def _calc_all_placement(self):
        all_device = self._calc_device()
        all_hierarchy = [
            (self.world_size,),
            (self.node_size, self.num_rank_for_each_node),
        ]
        return [
            flow.placement(device, np.array(range(self.world_size)).reshape(hierarchy))
            for device, hierarchy in list(product(all_device, all_hierarchy))
        ]

    def _calc_value(self):
        return self._calc_all_placement()


class all_cpu_placement(all_placement):
    def __init__(self):
        super().__init__()

    def _calc_device(self):
        return ["cpu"]


class all_cuda_placement(all_placement):
    def __init__(self):
        super().__init__()

    def _calc_device(self):
        return ["cuda"]


class random_placement(all_placement):
    def __init__(self):
        super().__init__()

    def _calc_value(self):
        return random_util.choice(self._calc_all_placement())


class random_cpu_placement(random_placement):
    def __init__(self):
        super().__init__()

    def _calc_device(self):
        return ["cpu"]


class random_gpu_placement(random_placement):
    def __init__(self):
        super().__init__()

    def _calc_device(self):
        return ["cuda"]


class all_sbp(generator):
    def __init__(
        self,
        placement=None,
        dim=1,
        max_dim=0,
        except_split=False,
        except_broadcast=False,
        except_partial_sum=False,
        valid_split_axis: Optional[Union[int, Sequence[int]]] = None,
    ):
        super().__init__([])
        if placement is not None:
            if isinstance(placement, random_placement):
                self.dim = len(placement.value().ranks.shape)
            elif isinstance(placement, flow.placement):
                self.dim = len(placement.ranks.shape)
            else:
                raise RuntimeError(
                    f"placement should be instance of random_placement or oneflow.placement"
                )
        else:
            self.dim = dim
        self.max_dim = max_dim
        self.except_split = except_split
        self.except_broadcast = except_broadcast
        self.except_partial_sum = except_partial_sum
        if valid_split_axis is not None:
            if isinstance(valid_split_axis, int):
                self.valid_split_axis = [
                    valid_split_axis,
                ]
            else:
                self.valid_split_axis = list(valid_split_axis)
        else:
            self.valid_split_axis = [i for i in range(self.max_dim)]

    def __len__(self):
        return len(self.value())

    def __getitem__(self, key):
        return self.value()[key]

    def _calc_all_sbp(self):
        # scalar only use broadcast sbp
        if self.max_dim == 0:
            return [
                [flow.sbp.broadcast for i in range(self.dim)],
            ]
        all_sbps = []
        if not self.except_split:
            for i in range(self.max_dim):
                if i in self.valid_split_axis:
                    all_sbps.append(flow.sbp.split(i))
        if not self.except_broadcast:
            all_sbps.append(flow.sbp.broadcast)
        if not self.except_partial_sum:
            all_sbps.append(flow.sbp.partial_sum)
        return list(product(all_sbps, repeat=self.dim))

    def _calc_value(self):
        return self._calc_all_sbp()


class random_sbp(all_sbp):
    def __init__(
        self,
        placement=None,
        dim=1,
        max_dim=0,
        except_split=False,
        except_broadcast=False,
        except_partial_sum=False,
        valid_split_axis: Optional[Union[int, Sequence[int]]] = None,
    ):
        super().__init__(
            placement,
            dim,
            max_dim,
            except_split,
            except_broadcast,
            except_partial_sum,
            valid_split_axis,
        )

    def _calc_value(self):
        return random_util.choice(self._calc_all_sbp())


@data_generator(torch.Tensor)
class choice_pytorch_tensor(generator):
    def __init__(self, a, size=None, replace=True, p=None, dtype=int):
        self.a = a
        self.size = size
        self.replace = replace
        self.p = p
        self.dtype = dtype
        super().__init__(
            [self.a, self.size, self.replace, self.p, self.dtype,]
        )

    def _calc_value(self):
        pytorch_tensor = None
        np_arr = np.random.choice(self.a, self.size, self.replace, self.p)
        torch_dtype = None
        return torch.tensor(np_arr.astype(self.dtype))


__all__ = [
    "random_pytorch_tensor",
    "random_bool",
    "random_device",
    "random_pytorch_dtype",
    "cpu_device",
    "gpu_device",
    "random_placement",
    "random_cpu_placement",
    "random_gpu_placement",
    "all_placement",
    "all_cpu_placement",
    "all_cuda_placement",
    "random_sbp",
    "all_sbp",
    "random",
    "random_or_nothing",
    "oneof",
    "constant",
    "nothing",
    "choice_pytorch_tensor",
]
