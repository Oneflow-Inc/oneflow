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

import os

import oneflow as flow
from typing import Union, Tuple, List, Optional, Sequence, Callable


def user_sigmoid(x, device_sub_tag="py", name: Optional[str] = None):
    return (
        flow.user_op_builder(
            name if name is not None else flow.util.unique_str("UserSigmoid_")
        )
        .Op("user_sigmoid")
        .Input("x", [x])
        .Output("y")
        .Attr("device_sub_tag", device_sub_tag)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def user_sigmoid_grad(
    y, dy, device_sub_tag="py", name: Optional[str] = None,
):
    return (
        flow.user_op_builder(
            name if name is not None else flow.util.unique_str("UerSigmoidGrad_")
        )
        .Op("user_sigmoid_grad")
        .Input("y", [y])
        .Input("dy", [dy])
        .Output("dx")
        .Attr("device_sub_tag", device_sub_tag)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
