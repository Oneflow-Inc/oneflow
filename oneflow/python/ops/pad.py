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

from typing import Optional, Sequence, Union

import oneflow
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow_api


@oneflow_export("pad")
def pad(
    x: oneflow_api.BlobDesc,
    paddings: Sequence[int],
    constant_value: Union[int, float] = 0,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator pads the input blob with constant value that user specifies. User can set the amount of padding by setting the parameter `paddings`. 

    Args:
        x (oneflow_api.BlobDesc): The input Blob
        paddings (Sequence[int]): A list of integers to specify the padding width, its length must equal with the length of `x.shape`. 
        constant_value (Union[int, float], optional): The constant value to pad. Defaults to 0.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Raises:
        ValueError: The parameter `paddings` must be a tuple or a list. 

    Returns:
        oneflow_api.BlobDesc: The Blob after padding.  

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np 


        @flow.global_function()
        def pad_Job(x: tp.Numpy.Placeholder((3, 3))
        ) -> tp.Numpy:
            return flow.pad(x, 
                            paddings=((2, 2), (1, 1)), 
                            constant_value=5)


        x = np.array([[1, 1, 1], 
                    [1, 1, 1], 
                    [1, 1, 1]]).astype(np.float32)
        out = pad_Job(x)

        # out [[5. 5. 5. 5. 5.]
        #      [5. 5. 5. 5. 5.]
        #      [5. 1. 1. 1. 5.]
        #      [5. 1. 1. 1. 5.]
        #      [5. 1. 1. 1. 5.]
        #      [5. 5. 5. 5. 5.]
        #      [5. 5. 5. 5. 5.]]

    """
    padding_before = []
    padding_after = []
    if isinstance(paddings, (list, tuple)):
        assert len(paddings) == len(x.shape), ValueError(
            "paddings must be the same size of input dims"
        )
        for p in paddings:
            assert isinstance(p, (list, tuple)) and len(p) == 2, ValueError(
                "the elem of paddings must be a tuple or a list with length of 2"
            )
            padding_before.append(p[0])
            padding_after.append(p[1])
    else:
        raise ValueError("paddings must be a tuple or a list.")
    if x.dtype in [
        oneflow.float32,
        oneflow.float16,
        oneflow.float64,
    ]:
        floating_constant_value = float(constant_value)
        integral_constant_value = int(0)
    else:
        floating_constant_value = float(0)
        integral_constant_value = int(constant_value)
    return (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("Pad_"))
        .Op("pad")
        .Input("x", [x])
        .Output("y")
        .Attr("padding_before", padding_before)
        .Attr("padding_after", padding_after)
        .Attr("floating_constant_value", floating_constant_value)
        .Attr("integral_constant_value", integral_constant_value)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("pad_grad")
def pad_grad(
    x: oneflow_api.BlobDesc,
    paddings: Sequence[int],
    constant_value: Union[int, float] = 0,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    padding_before = []
    padding_after = []
    if isinstance(paddings, (list, tuple)):
        assert len(paddings) == len(x.shape), ValueError(
            "paddings must be the same size of input dims"
        )
        for p in paddings:
            assert isinstance(p, (list, tuple)) and len(p) == 2, ValueError(
                "the elem of paddings must be a tuple or a list with length of 2"
            )
            padding_before.append(p[0])
            padding_after.append(p[1])
    else:
        raise ValueError("paddings must be a tuple or a list.")
    return (
        oneflow.user_op_builder(
            name if name is not None else id_util.UniqueStr("PadGrad_")
        )
        .Op("pad_grad")
        .Input("dy", [x])
        .Output("dx")
        .Attr("padding_before", padding_before)
        .Attr("padding_after", padding_after)
        .Attr("floating_constant_value", float(constant_value))
        .Attr("integral_constant_value", int(constant_value))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("same_padding")
def same_padding(
    x: oneflow_api.BlobDesc,
    padding: Sequence[int],
    data_format: str,
    kernel_size: Sequence[int],
    strides: Sequence[int],
    dilation_rate: Sequence[int],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator do the padding in "SAME" mode, It can computes the pad width according to the `kernel_size` and `strides` to keep the size of feature map unchanged after convolution or other operations. 

    Args:
        x (oneflow_api.BlobDesc): The input blob. 
        padding (Sequence[int]): The padding mode. It should be "SAME_UPPER" or "SAME_LOWER" 
        data_format ([type]): The data format of input Blob. If the string starts with "NC", it means the data format is `channel first`, else the data format is `channel last`. 
        kernel_size (Sequence[int]): The kernel size of operations. Its type should be tuple or list. 
        strides (Sequence[int]): The strides of operations. Its type should be tuple or list. 
        dilation_rate (Sequence[int]): The dilation rate of operations. Its type should be tuple or list.  
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The Blob after padding. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np 


        @flow.global_function()
        def same_pad_Job(x: tp.Numpy.Placeholder((1, 1, 3, 3))
        ) -> tp.Numpy:
            return flow.same_padding(x, 
                                    padding="SAME_UPPER", 
                                    data_format="NCHW", 
                                    kernel_size=(3, 3), 
                                    strides=(1, 1), 
                                    dilation_rate=(1, 1))


        x = np.ones(shape=(1, 1, 3, 3)).astype(np.float32)
        out = same_pad_Job(x)

        # out [[[[0. 0. 0. 0. 0.]
        #        [0. 1. 1. 1. 0.]
        #        [0. 1. 1. 1. 0.]
        #        [0. 1. 1. 1. 0.]
        #        [0. 0. 0. 0. 0.]]]]

    """
    assert isinstance(padding, str) and (
        padding.upper() == "SAME_LOWER" or padding.upper() == "SAME_UPPER"
    ), 'padding must be "SAME_LOWER" or "SAME_UPPER".'
    channel_pos = "channels_first" if data_format.startswith("NC") else "channels_last"
    assert isinstance(kernel_size, (list, tuple))
    assert isinstance(strides, (list, tuple))
    assert isinstance(dilation_rate, (list, tuple))
    num_spatial_dims = len(x.shape) - 2
    assert len(kernel_size) == num_spatial_dims
    assert len(strides) == num_spatial_dims
    assert len(dilation_rate) == num_spatial_dims

    return (
        oneflow.user_op_builder(
            name if name is not None else id_util.UniqueStr("SamePadding_")
        )
        .Op("same_padding")
        .Input("x", [x])
        .Output("y")
        .Attr("padding", padding.lower())
        .Attr("data_format", channel_pos)
        .Attr("kernel_size", kernel_size)
        .Attr("strides", strides)
        .Attr("dilation_rate", dilation_rate)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("reflection_pad2d")
def reflection_pad2d(
    x: oneflow_api.BlobDesc,
    padding: Union[int, tuple, list],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """Pads the input tensor using the reflection of the input boundary. 

    Args:
        x (oneflow_api.BlobDesc): input blob, only support "NCHW" format.
        padding (Union[int, oneflow_api.BlobDesc]): The size or bundary of padding, if is int uses the same padding in all dimension;
        if 4-dims tuple, uses (\text{padding\_left}padding_left , \text{padding\_right}padding_right , \text{padding\_top}padding_top , \text{padding\_bottom}padding_bottom )
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: [description]

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np 


        @flow.global_function()
        def pad_Job(x: tp.Numpy.Placeholder((1, 2, 3, 3))
        ) -> tp.Numpy:
            return flow.reflection_pad2d(x, padding=[2, 2, 1, 1])


        x = np.arange(18).reshape((1, 2, 3, 3)).astype(np.float32)
        out = pad_Job(x)

        # out [[[[ 5.  4.  3.  4.  5.  4.  3.]
        #    [ 2.  1.  0.  1.  2.  1.  0.]
        #    [ 5.  4.  3.  4.  5.  4.  3.]
        #    [ 8.  7.  6.  7.  8.  7.  6.]
        #    [ 5.  4.  3.  4.  5.  4.  3.]]

        #   [[ 14. 13. 12. 13. 14. 13. 12.]
        #    [ 11. 10.  9. 10. 11. 10.  9.]
        #    [ 14. 13. 12. 13. 14. 13. 12.]
        #    [ 17. 16. 15. 16. 17. 16. 15.]
        #    [ 14. 13. 12. 13. 14. 13. 12.]]]]

    """
    H, W = x.shape[2], x.shape[3]
    if isinstance(padding, (tuple, list)):
        assert len(padding) == len(x.shape), ValueError(
            "padding boundry must be the same size of input dims"
        )
        assert (
            padding[2] < H and padding[3] < H and padding[0] < W and padding[1] < W
        ), ValueError(
            "Padding size should be less than the corresponding input dimension!"
        )
        boundry = [padding[0], padding[1], padding[2], padding[3]]
    elif isinstance(padding, int):
        assert padding < H and padding < W, ValueError(
            "Padding size should be less than the corresponding input dimension!"
        )
        boundry = [padding, padding, padding, padding]
    else:
        raise ValueError("padding must be in or list or tuple!")

    return (
        oneflow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Reflection_Pad2d")
        )
        .Op("reflection_pad2d")
        .Input("x", [x])
        .Output("y")
        .Attr("padding", list(boundry))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("replication_pad2d")
def replication_pad2d(
    x: oneflow_api.BlobDesc,
    padding: Union[int, tuple, list],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """Pads the input tensor using the replication of the input boundary. 

    Args:
        x (oneflow_api.BlobDesc): input blob, only support "NCHW" format.
        padding (Union[int, oneflow_api.BlobDesc]): The size or bundary of padding, if is int uses the same padding in all dimension;
        if 4-dims tuple, uses (\text{padding\_left}padding_left , \text{padding\_right}padding_right , \text{padding\_top}padding_top , \text{padding\_bottom}padding_bottom )
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: [description]

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np 


        @flow.global_function()
        def pad_Job(x: tp.Numpy.Placeholder((1, 2, 3, 3))
        ) -> tp.Numpy:
            return flow.reflection_pad2d(x, padding=[2, 2, 1, 1])


        x = np.arange(18).reshape((1, 2, 3, 3)).astype(np.float32)
        out = pad_Job(x)

        # out [[[[ 0.  0.  0.  1.  2.  2.  2.]
        #    [ 0.  0.  0.  1.  2.  2.  2.]
        #    [ 3.  3.  3.  4.  5.  5.  5.]
        #    [ 6.  6.  6.  7.  8.  8.  8.]
        #    [ 6.  6.  6.  7.  8.  8.  8.]]

        #   [[ 9.  9.  9.  10.  11.  11.  11.]
        #    [ 9.  9.  9.  10.  11.  11.  11.]
        #    [ 12.  12.  12.  13.  14.  14.  14.]
        #    [ 15.  15.  15.  16.  17.  17.  17.]
        #    [ 15.  15.  15.  16.  17.  17.  17.]]]]

    """
    H, W = x.shape[2], x.shape[3]
    if isinstance(padding, (tuple, list)):
        assert len(padding) == len(x.shape), ValueError(
            "padding boundry must be the same size of input dims"
        )
        assert (
            padding[2] < H and padding[3] < H and padding[0] < W and padding[1] < W
        ), ValueError(
            "Padding size should be less than the corresponding input dimension!"
        )
        boundry = [padding[0], padding[1], padding[2], padding[3]]
    elif isinstance(padding, int):
        assert padding < H and padding < W, ValueError(
            "Padding size should be less than the corresponding input dimension!"
        )
        boundry = [padding, padding, padding, padding]
    else:
        raise ValueError("padding must be in or list or tuple!")

    return (
        oneflow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Replication_Pad2d")
        )
        .Op("replication_pad2d")
        .Input("x", [x])
        .Output("y")
        .Attr("padding", list(boundry))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("constant_pad2d")
def constant_pad2d(
    x: oneflow_api.BlobDesc,
    padding: Union[int, tuple, list],
    constant_value: Union[int, float] = 0,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """Pads the input tensor using an input constant value. 

    Args:
        x (oneflow_api.BlobDesc): input blob, only support "NCHW" format.
        padding (Union[int, oneflow_api.BlobDesc]): The size or bundary of padding, if is int uses the same padding in all dimension;
        if 4-dims tuple, uses (\text{padding\_left}padding_left , \text{padding\_right}padding_right , \text{padding\_top}padding_top , \text{padding\_bottom}padding_bottom )
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: [description]

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np 


        @flow.global_function()
        def pad_Job(x: tp.Numpy.Placeholder((1, 2, 3, 3), const_value)
        ) -> tp.Numpy:
            return flow.constant_pad2d(x, padding=[2, 2, 1, 1], const_value)


        x = np.arange(18).reshape((1, 2, 3, 3)).astype(np.float32)
        const_value = 1.5
        out = pad_Job(x, const_value)

        # out [[[[ 1.5  1.5  1.5  1.5  1.5  1.5  1.5]
        #    [ 1.5  1.5  0.  1.  2.  1.5  1.5]
        #    [ 1.5  1.5  3.  4.  5.  1.5  1.5]
        #    [ 1.5  1.5  6.  7.  8.  1.5  1.5]
        #    [ 1.5  1.5  1.5  1.5  1.5  1.5  1.5]]

        #   [[ 1.5  1.5  1.5  1.5  1.5  1.5  1.5.]
        #    [ 1.5  1.5  9.  10.  11.  1.5  1.5]
        #    [ 1.5  1.5  12.  13.  14.  1.5  1.5]
        #    [ 1.5  1.5  15.  16.  17.  1.5  1.5]
        #    [ 1.5  1.5  1.5  1.5  1.5  1.5  1.5]]]]

    """
    H, W = x.shape[2], x.shape[3]
    if isinstance(padding, (tuple, list)):
        assert len(padding) == len(x.shape), ValueError(
            "padding boundry must be the same size of input dims"
        )
        assert (
            padding[2] < H and padding[3] < H and padding[0] < W and padding[1] < W
        ), ValueError(
            "Padding size should be less than the corresponding input dimension!"
        )
        boundry = [padding[0], padding[1], padding[2], padding[3]]
    elif isinstance(padding, int):
        assert padding < H and padding < W, ValueError(
            "Padding size should be less than the corresponding input dimension!"
        )
        boundry = [padding, padding, padding, padding]
    else:
        raise ValueError("padding must be in or list or tuple!")

    if x.dtype in [
        oneflow.float32,
        oneflow.float16,
        oneflow.float64,
    ]:
        floating_value = float(constant_value)
        integral_value = int(0)
    else:
        floating_value = float(0)
        integral_value = int(constant_value)

    return (
        oneflow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Constant_Pad2d")
        )
        .Op("constant_pad2d")
        .Input("x", [x])
        .Output("y")
        .Attr("padding", list(boundry))
        .Attr("floating_value", floating_value)
        .Attr("integral_value", integral_value)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
