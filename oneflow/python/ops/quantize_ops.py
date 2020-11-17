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


@oneflow_export("quantization.MinMaxObserver")
def min_max_observer(
    input: remote_blob_util.BlobDef,
    quantize_to_bit: int = 8,
    quantize_scheme: str = "symmetric",
    per_layer_quantize: bool = True,
    name: Optional[str] = None,
) -> Tuple[remote_blob_util.BlobDef, remote_blob_util.BlobDef]:
    r"""Calculate the quantization scale and zero_point of the input `Blob`.

    Args:
        input (remote_blob_util.BlobDef): A `Blob`.
        quantize_to_bit (Optional[int], optional): Optional, int value. Quantize input to uintX / intX, X can be in range [2, 8]. Defaults to 8. 
        quantize_scheme (Optional[str], optional): Optional, str value. "symmetric" or "affine", quantize to signed / unsigned integer. Defaults to "symmetric". 
        per_layer_quantize (Optional[bool], optional): Optional, bool value. True or False, means per-layer / per-channel quantize. Defaults to True.
        name (Optional[str], optional):  This operator's name(optional). Defaults to None.

    Returns:
        Tuple[remote_blob_util.BlobDef, remote_blob_util.BlobDef]: The scale and zero_point of input `Blob`.
    
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
                scale, zero_point = flow.quantization.MinMaxObserver(
                    input, quantize_to_bit, quantize_scheme, per_layer_quantize
                )
            return scale, zero_point

        check_point = flow.train.CheckPoint()
        check_point.init()
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
        .Attr("quantize_to_bit", quantize_to_bit)
        .Attr("quantize_scheme", quantize_scheme)
        .Attr("per_layer_quantize", per_layer_quantize)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )

    return scale, zero_point


@oneflow_export("quantization.MovingAverageMinMaxObserver")
def moving_average_min_max_observer(
    input: remote_blob_util.BlobDef,
    quantize_to_bit: int = 8,
    quantize_scheme: str = "symmetric",
    momentum: float = 0.95,
    name: Optional[str] = None,
) -> Tuple[remote_blob_util.BlobDef, remote_blob_util.BlobDef]:

    op_name = (
        name if name is not None else id_util.UniqueStr("MovingAverageMinMaxObserver_")
    )

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
        .Attr("stop_update_after_iters", stop_update_after_iters)
        .Attr("quantize_to_bit", quantize_to_bit)
        .Attr("quantize_scheme", quantize_scheme)
        .Attr("momentum", momentum)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )

    return scale, zero_point


@oneflow_export("quantization.FakeQuantize")
def fake_quantization(
    input: remote_blob_util.BlobDef,
    scale: remote_blob_util.BlobDef,
    zero_point: remote_blob_util.BlobDef,
    quantize_to_bit: int = 8,
    quantize_scheme: str = "symmetric",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:

    builder = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Fake_Quantization_")
        )
        .Op("fake_quantization")
        .Input("in", [input])
        .Input("scale", [scale])
    )

    if quantize_scheme == "affine":
        builder = builder.Input("zero_point", [zero_point])

    out = (
        builder.Output("out")
        .Attr("quantize_to_bit", quantize_to_bit)
        .Build()
        .InferAndTryRun()
        .SoleOutputBlob()
    )

    return out
