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
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.modules.module import Module


class MinMaxObserver(Module):
    """
    
    Compute the quantization parameters of the input tensor.

    First compute the max and min values of input tensor:

    .. math::

        & max\\_value = max(input)

        & min\\_value = min(input)

    Then compute the scale and zero_point with the following equations:

        if quantization_scheme == "symmetric":

        .. math::

            & denom = 2^{quantization\\_to\\_bit - 1} - 1

            & scale = max(|max\\_value|,|min\\_value|) / denom

            & zero\\_point = 0

        elif quantization_scheme == "affine":

        .. math::

            & denom = 2^{quantization\\_to\\_bit} - 1

            & scale = (max\\_value - min\\_value) / denom

            & zero\\_point = -min\\_value / scale

    If per_layer_quantization is False, then the shape of scale and zero_point will be (input.shape[0],).

    Args:
        input(oneflow.Tensor):  the input value(s), in ``oneflow.float32``.
        quantization_formula (str): Support "google" or "cambricon".
        quantization_bit (int): Quantize input to uintX / intX, X can be in range [2, 8]. Defaults to 8.
        quantization_scheme (str): "symmetric" or "affine", quantize to signed / unsigned integer. Defaults to "symmetric".
        per_layer_quantization (bool): True or False, means per-layer / per-channel quantization. Defaults to True.

    Returns:
        Tuple[oneflow.Tensor, oneflow.Tensor]: The scale and zero_point of input tensor.

    For example:

    .. code-block:: python
        
        >>> import numpy as np
        >>> import oneflow as flow

        >>> weight = (np.random.random((2, 3, 4, 5)) - 0.5).astype(np.float32)
        
        >>> input_tensor = flow.tensor(
        ...    weight, dtype=flow.float32
        ... )
        
        >>> quantization_bit = 8
        >>> quantization_scheme = "symmetric"
        >>> quantization_formula = "google"
        >>> per_layer_quantization = True

        >>> min_max_observer = flow.nn.MinMaxObserver(quantization_formula=quantization_formula, quantization_bit=quantization_bit,
        ... quantization_scheme=quantization_scheme, per_layer_quantization=per_layer_quantization)

        >>> scale, zero_point = min_max_observer(
        ...    input_tensor, )

    """

    def __init__(
        self,
        quantization_formula: str = "google",
        quantization_bit: int = 8,
        quantization_scheme: str = "symmetric",
        per_layer_quantization: bool = True,
    ) -> None:
        super().__init__()
        self.quantization_formula = quantization_formula
        self.quantization_bit = quantization_bit
        self.quantization_scheme = quantization_scheme
        self.per_layer_quantization = per_layer_quantization

    def forward(self, input):
        return flow._C.min_max_observer(
            input,
            self.quantization_formula,
            self.quantization_bit,
            self.quantization_scheme,
            self.per_layer_quantization,
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
