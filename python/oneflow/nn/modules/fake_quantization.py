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
from oneflow.nn.module import Module


class FakeQuantization(Module):
    """
    """

    def __init__(
        self,
        quantization_formula: str = "google",
        quantization_bit: int = 8,
        quantization_scheme: str = "symmetric",
    ) -> None:
        super().__init__()
        self.quantization_formula = quantization_formula
        self.quantization_bit = quantization_bit
        self.quantization_scheme = quantization_scheme

    def forward(self, input, scale, zero_point):
        return flow.F.fake_quantization(
            input,
            scale,
            zero_point,
            self.quantization_formula,
            self.quantization_bit,
            self.quantization_scheme,
        )
