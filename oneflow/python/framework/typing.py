from __future__ import absolute_import

from typing import Sequence, Optional
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.input_blob_def as input_blob_def
import oneflow.python.framework.dtype as dtype_util


@oneflow_export("Numpy")
class Numpy:
    """`Numpy` is a type hint for numpy output of a OneFlow global function
    For instance::

        @oneflow.global_function()
        def foo() -> oneflow.Numpy:
            loss = ... # your network
            return loss
        
        loss = foo() # get a numpy.ndarray
        print(loss)
    """

    def Def(
        shape: Sequence[int], dtype=dtype_util.float, batch_axis: Optional[int] = 0
    ):
        """`Numpy.Def` is a typing function for numpy input of a OneFlow global function. 
        A `numpy.ndarray` takes a `Numpy.Def`'s place must have a identical shape.
        For instance::
            
            @oneflow.global_function()
            def foo(
                image_blob: oneflow.Numpy.Def(
                    (2, 255, 255, 3), dtype=flow.float32
                )
            ):
                # your network
            
            foo(np.random.randn(2, 255, 255, 3).astype(np.float32))
            
        """
        assert type(shape) is tuple, "shape should be a tuple. %s found" % shape
        return type(
            "Numpy.Def",
            (NumpyDef,),
            dict(shape=shape, dtype=dtype, batch_axis=batch_axis),
        )


@oneflow_export("List.Numpy")
class ListOfNumpy:
    """`List.Numpy` is a type hint for numpy output of a OneFlow global function
    For instance::

        @oneflow.global_function()
        def foo() -> oneflow.List.Numpy:
            mirrored_tensors = ... # your network
            return mirrored_tensors
        
        mirrored_tensors = foo() # get a list of numpy.ndarray
        for tensor in mirrored_tensors:
            print(mirrored_tensors)
    """

    def Def(
        shape: Sequence[int], dtype=dtype_util.float, batch_axis: Optional[int] = 0
    ):
        """`List.Numpy.Def` is a typing function for numpy input of a OneFlow global function. 
        A `list` of `numpy.ndarray` takes a `List.Numpy.Def`'s place. Each `numpy.ndarray` in the `list` could have any shape as long as it has the same rank and a smaller/equal size.
        For instance::
            
            @oneflow.global_function()
            def foo(
                image_blob: oneflow.List.Numpy.Def(
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
            "List.Numpy.Def",
            (ListOfNumpyDef,),
            dict(shape=shape, dtype=dtype, batch_axis=batch_axis),
        )


@oneflow_export("List.List.Numpy")
class ListOfListOfNumpy:
    """`List.List.Numpy` is a type hint for numpy output of a OneFlow global function
    For instance::

        @oneflow.global_function()
        def foo() -> oneflow.List.List.Numpy:
            mirrored_tensor_lists = ... # your network
            return mirrored_tensor_lists
        
        mirrored_tensor_lists = foo() # get a list of list of numpy.ndarray
        for tensor_list in mirrored_tensor_lists:
            for tensor in tensor_list:
                print(mirrored_tensors)
    """

    def Def(
        shape: Sequence[int], dtype=dtype_util.float, batch_axis: Optional[int] = 0
    ):
        """`List.List.Numpy.Def` is a typing function for numpy input of a OneFlow global function. 
        A `list` of `list` of `numpy.ndarray` takes a `List.List.Numpy.Def`'s place. Each `numpy.ndarray` in the `list` could have any shape as long as it has the same rank and a smaller/equal size.
        For instance::
            
            @oneflow.global_function()
            def foo(
                image_blob: oneflow.List.List.Numpy.Def(
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
            "List.List.Numpy.Def",
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
