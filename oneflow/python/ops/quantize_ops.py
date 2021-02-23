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
from typing import Tuple, Optional
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow_api


@oneflow_export("quantization.min_max_observer")
def min_max_observer(
    input: oneflow_api.BlobDesc,
    quantization_bit: int = 8,
    quantization_scheme: str = "symmetric",
    quantization_formula: str = "google",
    per_layer_quantization: bool = True,
    name: Optional[str] = None,
) -> Tuple[oneflow_api.BlobDesc, oneflow_api.BlobDesc]:
    r"""Compute the quantization parameters of the input tensor.

    First compute the max and min values of input tensor:

    .. math::

        & max\_value = max(input)

        & min\_value = min(input)

    Then compute the scale and zero_point with the following equations:

        if quantization_scheme == "symmetric": 

        .. math::

            & denom = 2^{quantization\_to\_bit - 1} - 1
            
            & scale = max(|max\_value|,|min\_value|) / denom

            & zero\_point = 0

        elif quantization_scheme == "affine":

        .. math::

            & denom = 2^{quantization\_to\_bit} - 1
            
            & scale = (max\_value - min\_value) / denom

            & zero\_point = -min\_value / scale
    
    If per_layer_quantization is False, then the shape of scale and zero_point will be (input.shape[0],).

    Args:
        input (oneflow_api.BlobDesc): input tensor.
        quantization_bit (int): Quantize input to uintX / intX, X can be in range [2, 8]. Defaults to 8. 
        quantization_scheme (str): "symmetric" or "affine", quantize to signed / unsigned integer. Defaults to "symmetric". 
        quantization_formula (str): Support "google" or "cambricon".
        per_layer_quantization (bool): True or False, means per-layer / per-channel quantization. Defaults to True.
        name (Optional[str]): This operator's name. Defaults to None.

    Returns:
        Tuple[oneflow_api.BlobDesc, oneflow_api.BlobDesc]: The scale and zero_point of input tensor.
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function(type="predict", function_config=flow.FunctionConfig())
        def QuantizeJob(
            input: tp.Numpy.Placeholder(input_shape, dtype=type_name_to_flow_type[dtype])
        ): tp.Numpy
            with flow.scope.placement(device_type, "0:0"):
                scale, zero_point = flow.quantization.min_max_observer(
                    input, quantization_bit=8,
                    quantization_scheme="symmetric",
                    quantization_formula="google",
                    per_layer_quantization=True
                )
            return scale, zero_point

        input = (np.random.random(input_shape) - 0.5).astype(type_name_to_np_type[dtype])
        scale, zero_point = QuantizeJob(input)

    """
    scale, zero_point = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("MinMaxObserver_")
        )
        .Op("min_max_observer")
        .Input("in", [input])
        .Output("scale")
        .Output("zero_point")
        .Attr("quantization_bit", quantization_bit)
        .Attr("quantization_scheme", quantization_scheme)
        .Attr("quantization_formula", quantization_formula)
        .Attr("per_layer_quantization", per_layer_quantization)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )

    return scale, zero_point


@oneflow_export("quantization.moving_average_min_maxObserver")
def moving_average_min_max_observer(
    input: oneflow_api.BlobDesc,
    quantization_bit: int = 8,
    quantization_scheme: str = "symmetric",
    quantization_formula: str = "google",
    momentum: float = 0.95,
    name: Optional[str] = None,
) -> Tuple[oneflow_api.BlobDesc, oneflow_api.BlobDesc]:
    r"""Compute the quantization parameters based on the moving average of the input tensor's min and max values.

    First compute the moving\_max and moving\_min value of input tensor:

        if quantization_scheme == "symmetric": 

        .. math::

            & moving\_max = moving\_max * momentum + |max(input)| * (1 - momentum)

            & moving\_min = moving\_max

        elif quantization_scheme == "affine":

        .. math::

            & moving\_max = moving\_max * momentum + max(input) * (1 - momentum)

            & moving\_min = moving\_min * momentum + min(input) * (1 - momentum)

    The moving average of min and max values are initialized as the first batch of input `Blob`'s min and max.

    Then compute the scale and zero_point with the following equations:

        if quantization_scheme == "symmetric": 

        .. math::

            & denom = 2^{quantization\_to\_bit - 1} - 1
            
            & scale = moving\_max / denom

            & zero\_point = 0

        elif quantization_scheme == "affine":

        .. math::

            & denom = 2^{quantization\_to\_bit} - 1
            
            & scale = (moving\_max - moving\_min) / denom

            & zero\_point = -moving\_min / scale
    
    Args:
        input (oneflow_api.BlobDesc): input tensor.
        quantization_bit (int): Quantize input to uintX / intX, X can be in range [2, 8]. Defaults to 8. 
        quantization_scheme (str): "symmetric" or "affine", quantize to signed / unsigned integer. Defaults to "symmetric". 
        quantization_formula (str): Support "google" or "cambricon".
        momentum (float): Smoothing parameter for exponential moving average operation. Defaults to 0.95.
        name (Optional[str]): This operator's name. Defaults to None.

    Returns:
        Tuple[oneflow_api.BlobDesc, oneflow_api.BlobDesc]: The scale and zero_point of input tensor.
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function(type="predict", function_config=flow.FunctionConfig())
        def QuantizeJob(
            input: tp.Numpy.Placeholder(input_shape, dtype=type_name_to_flow_type[dtype])
        ): tp.Numpy
            with flow.scope.placement(device_type, "0:0"):
                scale, zero_point = flow.quantization.moving_average_min_maxObserver(
                    input, quantization_bit=8,
                    quantization_scheme="symmetric",
                    quantization_formula="google",
                    momentum=0.95
                )
            return scale, zero_point

        input = (np.random.random(input_shape) - 0.5).astype(type_name_to_np_type[dtype])
        scale, zero_point = QuantizeJob(input)

    """
    op_name = (
        name if name is not None else id_util.UniqueStr("MovingAverageMinMaxObserver_")
    )

    training = True if flow.current_global_function_desc().IsTrainable() else False

    with flow.scope.namespace(op_name):
        moving_max = flow.get_variable(
            "moving_max",
            shape=(1,),
            dtype=input.dtype,
            initializer=flow.zeros_initializer(input.dtype),
            trainable=False,
        )
        moving_min = flow.get_variable(
            "moving_min",
            shape=(1,),
            dtype=input.dtype,
            initializer=flow.zeros_initializer(input.dtype),
            trainable=False,
        )
        current_train_step = flow.get_variable(
            "current_train_step",
            shape=(1,),
            dtype=flow.int64,
            initializer=flow.zeros_initializer(flow.int64),
            trainable=False,
        )
    stop_update_after_iters = 1
    scale, zero_point = (
        flow.user_op_builder(op_name)
        .Op("moving_average_min_max_observer")
        .Input("in", [input])
        .Input("current_train_step", [current_train_step])
        .Input("moving_max", [moving_max])
        .Input("moving_min", [moving_min])
        .Output("scale")
        .Output("zero_point")
        .Attr("training", training)
        .Attr("stop_update_after_iters", stop_update_after_iters)
        .Attr("quantization_bit", quantization_bit)
        .Attr("quantization_scheme", quantization_scheme)
        .Attr("quantization_formula", quantization_formula)
        .Attr("momentum", momentum)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )

    return scale, zero_point


@oneflow_export("quantization.fake_quantization")
def fake_quantization(
    input: oneflow_api.BlobDesc,
    scale: oneflow_api.BlobDesc,
    zero_point: oneflow_api.BlobDesc,
    quantization_bit: int = 8,
    quantization_scheme: str = "symmetric",
    quantization_formula: str = "google",
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Simulate the quantize and dequantize operations in training time.

    The output will be computed as:

        if quantization_scheme == "symmetric": 

        .. math::

            & quant\_max = 2^{quantization\_to\_bit - 1} - 1
            
            & quant\_min = -quant\_max

            & clamp(round(x / scale), quant\_min, quant\_max) * scale

        elif quantization_scheme == "affine":

        .. math::

            & quant\_max = 2^{quantization\_to\_bit} - 1
            
            & quant\_min = 0

            & (clamp(round(x / scale + zero\_point), quant\_min, quant\_max) - zero\_point) * scale

    Args:
        input (oneflow_api.BlobDesc): input tensor.
        scale (oneflow_api.BlobDesc): Computed by min_max_observer or moving_average_min_maxObserver op.
        zero_point (oneflow_api.BlobDesc): Computed by min_max_observer or moving_average_min_maxObserver op.
        quantization_bit (int): Quantize input to uintX / intX, X can be in range [2, 8]. Defaults to 8. 
        quantization_scheme (str): "symmetric" or "affine", quantize to signed / unsigned integer. Defaults to "symmetric". 
        quantization_formula (str): Support "google" or "cambricon".
        name (Optional[str]): This operator's name. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: Input tensor after quantize and dequantize operations.

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function(type="predict", function_config=flow.FunctionConfig())
        def QuantizeJob(
            input: tp.Numpy.Placeholder(input_shape, dtype=type_name_to_flow_type[dtype])
        ): tp.Numpy
            with flow.scope.placement(device_type, "0:0"):
                scale, zero_point = flow.quantization.min_max_observer(
                    input, quantization_bit=8,
                    quantization_scheme="symmetric",
                    quantization_formula="google",
                    per_layer_quantization=True
                )
                fake_quantize_out = flow.quantization.fake_quantization(
                    input, scale, zero_point,
                    quantization_bit=8,
                    quantization_scheme="symmetric",
                    quantization_formula="google"
                )
            return fake_quantize_out

        input = (np.random.random(input_shape) - 0.5).astype(type_name_to_np_type[dtype])
        fake_quantize_out = QuantizeJob(input)

    """
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Fake_Quantization_")
        )
        .Op("fake_quantization")
        .Input("in", [input])
        .Input("scale", [scale])
        .Input("zero_point", [zero_point])
        .Output("out")
        .Attr("quantization_bit", quantization_bit)
        .Attr("quantization_scheme", quantization_scheme)
        .Attr("quantization_formula", quantization_formula)
        .Build()
        .InferAndTryRun()
        .SoleOutputBlob()
    )
