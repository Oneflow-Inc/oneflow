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


@oneflow_export("nn.generate_quantize_scale_for_weight")
def generate_quantize_scale_for_weight(
    weight: remote_blob_util.BlobDef,
    quantize_to_bit: int = 8,
    quantizer_type: str = "symmetric",
    per_layer_quantization: bool = True,
    name: Optional[str] = None,
) -> Tuple[remote_blob_util.BlobDef, remote_blob_util.BlobDef]:

    scale, zero_point = (
        flow.user_op_builder(
            name
            if name is not None
            else id_util.UniqueStr("Generate_Quantize_Scale_For_Weight_")
        )
        .Op("generate_quantize_scale_for_weight")
        .Input("weight", [weight])
        .Output("scale")
        .Output("zero_point")
        .Attr("quantize_to_bit", quantize_to_bit)
        .Attr("quantizer_type", quantizer_type)
        .Attr("per_layer_quantization", per_layer_quantization)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )

    return scale, zero_point


@oneflow_export("nn.generate_quantize_scale_for_activation")
def generate_quantize_scale_for_activation(
    activation: remote_blob_util.BlobDef,
    moving_max: remote_blob_util.BlobDef,
    moving_min: remote_blob_util.BlobDef,
    quantize_to_bit: int = 8,
    quantizer_type: str = "symmetric",
    momentum: float = 0.95,
    name: Optional[str] = None,
) -> Tuple[remote_blob_util.BlobDef, remote_blob_util.BlobDef]:

    scale, zero_point = (
        flow.user_op_builder(
            name
            if name is not None
            else id_util.UniqueStr("Generate_Quantize_Scale_For_Activation_")
        )
        .Op("generate_quantize_scale_for_activation")
        .Input("activation", [activation])
        .Input("moving_max", [moving_max])
        .Input("moving_min", [moving_min])
        .Output("scale")
        .Output("zero_point")
        .Attr("quantize_to_bit", quantize_to_bit)
        .Attr("quantizer_type", quantizer_type)
        .Attr("momentum", momentum)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )

    return scale, zero_point


@oneflow_export("nn.fake_quantization")
def fake_quantization(
    input: remote_blob_util.BlobDef,
    scale: remote_blob_util.BlobDef,
    zero_point: remote_blob_util.BlobDef,
    quantize_to_bit: int = 8,
    quantizer_type: str = "symmetric",
    name: Optional[str] = None,
) -> Tuple[remote_blob_util.BlobDef, remote_blob_util.BlobDef]:

    out = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Fake_Quantization_")
        )
        .Op("fake_quantization")
        .Input("in", [input])
        .Input("scale", [scale])
        .Input("zero_point", [zero_point])
        .Output("out")
        .Attr("quantize_to_bit", quantize_to_bit)
        .Attr("quantizer_type", quantizer_type)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )

    return out
