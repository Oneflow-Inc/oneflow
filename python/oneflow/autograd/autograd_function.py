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

from oneflow._oneflow_internal import TensorTuple
from oneflow._oneflow_internal.autograd import AutogradFunctionBase


class Function(AutogradFunctionBase):
    def __init__(self):
        super().__init__(self.forward, self.backward)

    def __call__(self, *inputs):
        return self.apply(*inputs)

    def apply(self, *inputs):
        return super().apply(TensorTuple(inputs))

    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError("You must implement the forward function for custom autograd.Function.")

    @staticmethod
    def backward(ctx, *out_grads):
        raise NotImplementedError("You must implement the backward function for custom autograd.Function.")
