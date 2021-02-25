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
from __future__ import absolute_import

from typing import Sequence, Optional
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.input_blob_def as input_blob_def
import oneflow.python.framework.dtype as dtype_util
import typing
import inspect
import sys


class PyStructCompatibleToBlob(object):
    pass


@oneflow_export("typing.Numpy")
class Numpy(PyStructCompatibleToBlob):
    """`Numpy` is a type hint for numpy output of a OneFlow global function
    For instance::

        @oneflow.global_function()
        def foo() -> oneflow.typing.Numpy:
            loss = ... # your network
            return loss

        loss = foo() # get a numpy.ndarray
        print(loss)
    """

    def Placeholder(shape: Sequence[int], dtype=dtype_util.float):
        """`Numpy.Placeholder` is a typing function for numpy input of a OneFlow global function.
        A `numpy.ndarray` takes a `Numpy.Placeholder`'s place must have an identical shape.
        For instance::

            @oneflow.global_function()
            def foo(
                image_blob: oneflow.typing.Numpy.Placeholder(
                    (2, 255, 255, 3), dtype=flow.float32
                )
            ):
                # your network

            foo(np.random.randn(2, 255, 255, 3).astype(np.float32))

        """
        assert type(shape) is tuple, "shape should be a tuple. %s found" % shape
        return type("Numpy.Placeholder", (NumpyDef,), dict(shape=shape, dtype=dtype),)


@oneflow_export("typing.ListNumpy")
class ListNumpy(PyStructCompatibleToBlob):
    """`ListNumpy` is a type hint for numpy output of a OneFlow global function
    For instance::

        @oneflow.global_function()
        def foo() -> oneflow.typing.ListNumpy:
            mirrored_tensors = ... # your network
            return mirrored_tensors

        mirrored_tensors = foo() # get a list of numpy.ndarray
        for tensor in mirrored_tensors:
            print(mirrored_tensors)
    """

    def Placeholder(shape: Sequence[int], dtype=dtype_util.float):
        """`ListNumpy.Placeholder` is a typing function for numpy input of a OneFlow global function.
        A `list` of `numpy.ndarray` takes a `ListNumpy.Placeholder`'s place. Each `numpy.ndarray` in the `list` could have any shape as long as it has the same rank and a smaller/equal size.
        For instance::

            @oneflow.global_function()
            def foo(
                image_blob: oneflow.typing.ListNumpy.Placeholder(
                    (2, 255, 255, 3), dtype=flow.float32
                )
            ):
                # your network

            input1 = np.random.randn(2, 255, 255, 3).astype(np.float32)
            input2 = np.random.randn(2, 251, 251, 3).astype(np.float32)
            foo([input1])
            foo([input2])

        """
        assert type(shape) is tuple, "shape should be a tuple. %s found" % shape
        return type(
            "ListNumpy.Placeholder", (ListOfNumpyDef,), dict(shape=shape, dtype=dtype),
        )


@oneflow_export("typing.ListListNumpy")
class ListListNumpy(PyStructCompatibleToBlob):
    """`ListListNumpy` is a type hint for numpy output of a OneFlow global function
    For instance::

        @oneflow.global_function()
        def foo() -> oneflow.typing.ListListNumpy:
            mirrored_tensor_lists = ... # your network
            return mirrored_tensor_lists

        mirrored_tensor_lists = foo() # get a list of list of numpy.ndarray
        for tensor_list in mirrored_tensor_lists:
            for tensor in tensor_list:
                print(mirrored_tensors)
    """

    def Placeholder(shape: Sequence[int], dtype=dtype_util.float):
        """`ListListNumpy.Placeholder` is a typing function for numpy input of a OneFlow global function.
        A `list` of `list` of `numpy.ndarray` takes a `ListListNumpy.Placeholder`'s place. Each `numpy.ndarray` in the `list` could have any shape as long as it has the same rank and a smaller/equal size.
        For instance::

            @oneflow.global_function()
            def foo(
                image_blob: oneflow.typing.ListListNumpy.Placeholder(
                    (2, 255, 255, 3), dtype=flow.float32
                )
            ):
                # your network

            input1 = np.random.randn(2, 255, 255, 3).astype(np.float32)
            input2 = np.random.randn(2, 251, 251, 3).astype(np.float32)
            foo([[input1]])
            foo([[input2]])

        """
        assert type(shape) is tuple, "shape should be a tuple. %s found" % shape
        return type(
            "ListListNumpy.Placeholder",
            (ListOfListOfNumpyDef,),
            dict(shape=shape, dtype=dtype),
        )


class OneflowNumpyDef(object):
    @classmethod
    def NewInputBlobDef(subclass):
        raise NotImplementedError


class NumpyDef(OneflowNumpyDef):
    @classmethod
    def NewInputBlobDef(subclass):
        return input_blob_def.FixedTensorDef(subclass.shape, dtype=subclass.dtype)


class ListOfNumpyDef(OneflowNumpyDef):
    @classmethod
    def NewInputBlobDef(subclass):
        return input_blob_def.MirroredTensorDef(subclass.shape, dtype=subclass.dtype)


class ListOfListOfNumpyDef(OneflowNumpyDef):
    @classmethod
    def NewInputBlobDef(subclass):
        return input_blob_def.MirroredTensorListDef(
            subclass.shape, dtype=subclass.dtype
        )


@oneflow_export("typing.Callback")
class Callback(typing.Generic[typing.TypeVar("T")]):
    pass


@oneflow_export("typing.Bundle")
class Bundle(typing.Generic[typing.TypeVar("T")]):
    """
    One or a collection of  typing.Numpy/typing.ListNumpy/typing.ListListNumpy,
    such as x, [x], (x,), {"key": x} and the mixed form of them.
    """

    pass


def OriginFrom(parameterised, generic):
    if inspect.isclass(parameterised) and inspect.isclass(generic):
        return issubclass(parameterised, generic)
    if generic == OneflowNumpyDef:
        assert not inspect.isclass(parameterised)
        return False
    if (sys.version_info.major, sys.version_info.minor) >= (3, 7):
        if not hasattr(parameterised, "__origin__"):
            return False
        if generic == typing.Dict:
            return parameterised.__origin__ is dict
        if generic == typing.Tuple:
            return parameterised.__origin__ is tuple
        if generic == typing.List:
            return parameterised.__origin__ is list
        if generic == Callback:
            return parameterised.__origin__ is Callback
        if generic == Bundle:
            return parameterised.__origin__ is Bundle

    raise NotImplementedError("python typing is a monster torturing everyone.")
