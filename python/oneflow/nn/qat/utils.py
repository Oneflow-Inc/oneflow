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
import oneflow.nn as nn


def init_qat_helper_modules(
    self,
    quantization_formula: str = "google",
    quantization_bit: int = 8,
    quantization_scheme: str = "symmetric",
    weight_per_layer_quantization: bool = True,
    input_quant_momentum: float = 0.95,
):
    self.input_min_max_observer = nn.MovingAverageMinMaxObserver(
        stop_update_after_iters=1,
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        momentum=input_quant_momentum,
    )
    self.register_buffer("current_train_step", flow.zeros(1, dtype=flow.int64,))
    self.weight_min_max_observer = nn.MinMaxObserver(
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        per_layer_quantization=weight_per_layer_quantization,
    )
    self.fake_quantizer = nn.FakeQuantization(
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
    )
