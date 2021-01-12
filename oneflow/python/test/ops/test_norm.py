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
import numpy as np
import oneflow.typing as tp
from test_util import GenArgList
import unittest
from collections import OrderedDict
from typing import Dict
import os


def np_norm(x, p="fro", axis=None, keepdim=False, name=None):
    print("Np axis: ", axis)

    def vector_norm(input, porder=None, axis=None, keepdim=False, name=None):
        porder = float(porder) if porder is not None else 2.0

        if porder == np.inf:
            out = np.max(np.abs(input), axis=axis, keepdims=keepdim)
            out_broadcasted = np.broadcast_to(out, input.shape)
            if axis == 1:
                out_broadcasted = np.transpose(out_broadcasted, [1, 0])
            grad = (np.abs(input) == out_broadcasted) * np.sign(input)
            return out, grad

        elif porder == -np.inf:
            out = np.min(np.abs(input), axis=axis, keepdims=keepdim)
            out_broadcasted = np.broadcast_to(out, input.shape)
            if axis == 1:
                out_broadcasted = np.transpose(out_broadcasted, [1, 0])
            grad = (np.abs(input) == out_broadcasted) * np.sign(input)
            return out, grad
        elif porder == 0:
            # Zero norm
            out = np.sum((input != 0), keepdims=keepdim)
            grad = np.ones_like(input != 0)
            return out, grad
        else:
            abs_x = np.abs(input)
            pow_x = np.power(abs_x, porder)
            sum_x = np.sum(pow_x, axis=axis, keepdims=keepdim)
            out = np.power(sum_x, 1.0 / porder)
            grad = (
                np.sign(input)
                * np.power(np.abs(input), porder - 1)
                / (np.power(np.broadcast_to(out, input.shape), porder - 1))
            )
            return out, grad

    def frobenius_norm(input, axis=None, keepdim=False, name=None):
        if axis is not None and not (isinstance(axis, tuple) and len(axis) == 2):
            raise ValueError(
                "The dim of frobenius norm op should be None or two elements list!"
            )
        out = np.sqrt(np.sum(np.square(input), axis=axis, keepdims=keepdim))
        grad = input / np.broadcast_to(out, shape=input.shape)
        return out, grad

    def inf_norm(input, porder=np.inf, axis=None, keepdim=False, name=None):
        axis = axis if axis != None and axis != [] else [0]
        if porder == np.inf:
            out = np.max(np.abs(input), axis=axis, keepdims=keepdim)
            print("Inf broadcast", np.broadcast_to(out, input.shape))
            grad = (np.abs(input) == np.broadcast_to(out, input.shape)) * np.sign(input)
            return out, grad
        elif porder == -np.inf:
            out = np.min(np.abs(input), axis=axis, keepdims=keepdim)
            grad = (np.abs(input) == np.broadcast_to(out, input.shape)) * np.sign(input)
            return out, grad

    def p_matrix_norm(input, porder, axis, keepdim=False, name=None):
        abs_x = np.abs(input)
        out = np.power(abs_x, porder)
        out = np.sum(out, axis=axis, keepdims=keepdim)
        out = np.power(out, 1.0 / porder)
        grad = (
            np.sign(input)
            * np.power(np.abs(input), porder - 1)
            / (np.power(np.broadcast_to(out, input.shape), porder - 1))
        )

        return out, grad

    if axis is None and p is not None:
        if isinstance(p, str):
            if p == "fro":
                return frobenius_norm(x, axis=axis, keepdim=keepdim, name=name)
            else:
                raise ValueError(
                    "only valid string values are 'fro', found {}".format(p)
                )
        elif isinstance(p, (int, float)):
            return vector_norm(x, porder=p, axis=axis, keepdim=keepdim, name=name)

    if isinstance(axis, tuple) and len(axis) == 1:
        axis = axis[0]

    # calculate vector norm, where axis is int or list with only one integer
    if isinstance(axis, int):
        if isinstance(p, str):
            if p == "fro":
                return vector_norm(x, porder=2, axis=axis, keepdim=keepdim, name=name)

            else:
                raise ValueError(
                    "only valid string values are 'fro', found {}".format(p)
                )
        elif isinstance(p, (int, float)):
            return vector_norm(x, axis=axis, porder=p, keepdim=keepdim, name=name)
        else:
            raise ValueError(
                "unspport p for p-order vector norm. except float, found {}".format(p)
            )
    # calculate matrix norm, where axis is list with two integers
    elif isinstance(axis, tuple) and len(axis) == 2:
        if p == "fro":
            return frobenius_norm(x, axis=axis, keepdim=keepdim, name=name)
        elif p == np.inf or p == -np.inf:
            return inf_norm(x, porder=p, axis=axis, keepdim=keepdim, name=name)
        elif p == 0:
            raise ValueError(
                "just suport axis type int or list (length of list <=1) if p = 0, found {}".format(
                    axis
                )
            )
        else:
            return p_matrix_norm(x, porder=p, axis=axis, keepdim=keepdim, name=name)
    else:
        raise ValueError(
            "except axis type int or list (length of list <=2), found {}".format(axis)
        )


def _gen_arg_dict(shape, p=None, axis=None):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["x"] = [np.random.randn(*shape)]
    arg_dict["p"] = [*p]
    arg_dict["axis"] = [*axis]
    return arg_dict


def _gen_arg_dict_inf(shape, axis):
    arg_dict = _gen_arg_dict(shape=shape, axis=axis, p=[np.inf, -np.inf])
    return arg_dict


def _gen_arg_dict_fro(shape, axis):
    arg_dict = _gen_arg_dict(shape=shape, axis=axis, p=["fro"])
    return arg_dict


def _gen_arg_dict_other(shape, axis):
    arg_dict = _gen_arg_dict(shape=shape, axis=axis, p=[-2, -1, 1, 2, 3, -3])
    return arg_dict
