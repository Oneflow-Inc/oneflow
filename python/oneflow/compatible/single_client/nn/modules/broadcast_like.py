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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.nn.module import Module


class BroadCastLike(Module):
    def __init__(self, broadcast_axes: None) -> None:
        super().__init__()
        self.broadcast_axes = broadcast_axes

    def forward(self, x, like_tensor):
        return flow.F.broadcast_like(x, like_tensor, broadcast_axes=self.broadcast_axes)


def broadcast_like_op(x, like_tensor, broadcast_axes: None):
    return BroadCastLike(broadcast_axes=broadcast_axes)(x, like_tensor)
