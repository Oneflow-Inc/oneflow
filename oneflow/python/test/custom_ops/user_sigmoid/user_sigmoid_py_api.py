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


def user_sigmoid_forward(x, name: Optional[str] = None):
    return (
        flow.user_op_builder(
            name if name is not None else flow.util.unique_str("UserSigmoidForward_")
        )
        .Op("user_sigmoid_forward")
        .Input("x", [x])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def user_sigmoid_backward(
    y, dy, name: Optional[str] = None,
):
    return (
        flow.user_op_builder(
            name if name is not None else flow.util.unique_str("UerSigmoidBackward_")
        )
        .Op("user_sigmoid_backward")
        .Input("y", [y])
        .Input("dy", [dy])
        .Output("dx")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
