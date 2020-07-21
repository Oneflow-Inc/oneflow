from __future__ import absolute_import

from typing import Sequence, Optional
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.input_blob_def as input_blob_def
import oneflow.python.framework.dtype as dtype_util
import typing
import inspect
import sys


@oneflow_export("typing.Numpy")
class Numpy:
    """`Numpy` is a type hint for numpy output of a OneFlow global function
    For instance::

        @oneflow.global_function()
        def foo() -> oneflow.typing.Numpy:
            loss = ... # your network
            return loss
        
        loss = foo() # get a numpy.ndarray
        print(loss)
    """

    def Placeholder(
        shape: Sequence[int], dtype=dtype_util.float, batch_axis: Optional[int] = 0
    ):
        """`Numpy.Placeholder` is a typing function for numpy input of a OneFlow global function. 
        A `numpy.ndarray` takes a `Numpy.Placeholder`'s place must have a identical shape.
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
        return type(
            "Numpy.Placeholder",
            (NumpyDef,),
            dict(shape=shape, dtype=dtype, batch_axis=batch_axis),
        )


@oneflow_export("typing.ListNumpy")
class ListOfNumpy:
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

    def Placeholder(
        shape: Sequence[int], dtype=dtype_util.float, batch_axis: Optional[int] = 0
    ):
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
            "ListNumpy.Placeholder",
            (ListOfNumpyDef,),
            dict(shape=shape, dtype=dtype, batch_axis=batch_axis),
        )


@oneflow_export("typing.ListListNumpy")
class ListOfListOfNumpy:
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

    def Placeholder(
        shape: Sequence[int], dtype=dtype_util.float, batch_axis: Optional[int] = 0
    ):
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
            dict(shape=shape, dtype=dtype, batch_axis=batch_axis),
        )


class OneflowNumpyDef(object):
    @classmethod
    def NewInputBlobDef(subclass):
        raise NotImplementedError


class NumpyDef(OneflowNumpyDef):
    @classmethod
    def NewInputBlobDef(subclass):
        return input_blob_def.FixedTensorDef(
            subclass.shape, dtype=subclass.dtype, batch_axis=subclass.batch_axis
        )


class ListOfNumpyDef(OneflowNumpyDef):
    @classmethod
    def NewInputBlobDef(subclass):
        return input_blob_def.MirroredTensorDef(
            subclass.shape, dtype=subclass.dtype, batch_axis=subclass.batch_axis
        )


class ListOfListOfNumpyDef(OneflowNumpyDef):
    @classmethod
    def NewInputBlobDef(subclass):
        return input_blob_def.MirroredTensorListDef(
            subclass.shape, dtype=subclass.dtype, batch_axis=subclass.batch_axis
        )


def OriginFrom(parameterised, generic):
    if inspect.isclass(parameterised) and inspect.isclass(generic):
        return issubclass(parameterised, generic)
    if inspect.isclass(parameterised) != inspect.isclass(generic):
        return False
    if (sys.version_info.major, sys.version_info.minor) >= (3, 7):
        if not hasattr(parameterised, "__origin__"):
            return False
        if generic == typing.Tuple:
            return (
                type(parameterised) is type(typing.Tuple[int])
                and parameterised.__origin__ is tuple
            )
        if generic == typing.List:
            return (
                type(parameterised) is type(typing.List[int])
                and parameterised.__origin__ is list
            )
    raise NotImplementedError("python typing is a monster torturing everyone.")
