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


@oneflow_export("pad")
def pad(
    x: remote_blob_util.BlobDef,
    paddings: Sequence[int],
    constant_value: Union[int, float] = 0,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
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
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("Pad_"))
        .Op("pad")
        .Input("x", [x])
        .Output("y")
        .Attr("padding_before", padding_before)
        .Attr("padding_after", padding_after)
        .Attr("floating_constant_value", float(constant_value))
        .Attr("integral_constant_value", int(constant_value))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("pad_grad")
def pad_grad(
    x: remote_blob_util.BlobDef,
    paddings: Sequence[int],
    constant_value: Union[int, float] = 0,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
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
