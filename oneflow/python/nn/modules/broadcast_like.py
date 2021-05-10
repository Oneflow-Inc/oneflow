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
import oneflow as flow
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op


class BroadCastLike(Module):
    def __init__(self, broadcast_axes: None) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("broadcast_like")
            .Input("x")
            .Input("like")
            .Attr("broadcast_axes", broadcast_axes)
            .Output("y")
            .Build()
        )
        self.broadcast_axes = broadcast_axes

    def forward(self, x, like_tensor):
        return self._op(x, like_tensor, broadcast_axes=self.broadcast_axes)[0]


@oneflow_export("tmp.broadcast_like")
def broadcast_like_op(x, like_tensor, broadcast_axes: None):
    return BroadCastLike(broadcast_axes=broadcast_axes)(x, like_tensor)
