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


class Where(Module):
    def __init__(self) -> None:
        super().__init__()
        self._where_op = (
            flow.builtin_op("where")
            .Input("condition")
            .Input("x")
            .Input("y")
            .Output("out")
            .Build()
        )

    def forward(self, condition, x, y):
        if x.shape == condition.shape and y.shape == condition.shape:
            broadcast_cond = condition
            broadcast_x = x
            broadcast_y = y
        else:
            broadcast_like_shape = []
            broadcast_condition_axes = []
            broadcast_x_axes = []
            broadcast_y_axes = []
            
            for i in range(len(x.shape)):
                max_dim = max(x.shape[i], max(y.shape[i], condition.shape[i]))
                broadcast_like_shape.append(max_dim)
                if max_dim != condition.shape[i]:
                    broadcast_condition_axes.append(i)
                if max_dim != x.shape[i]:
                    broadcast_x_axes.append(i)
                if max_dim != y.shape[i]:
                    broadcast_y_axes.append(i)

            broadcast_like_tensor = flow.Tensor(broadcast_like_shape)
            broadcast_cond = flow.tmp.broadcast_like(condition, broadcast_like_tensor, broadcast_condition_axes)
            broadcast_x = flow.tmp.broadcast_like(x, broadcast_like_tensor, broadcast_x_axes)
            broadcast_y = flow.tmp.broadcast_like(y, broadcast_like_tensor, broadcast_y_axes)

        return self._where_op(broadcast_cond, broadcast_x, broadcast_y)[0]


@oneflow_export("tmp.where")
@register_tensor_op("where")
def where_op(condition, x, y):
    return Where()(condition, x, y)
