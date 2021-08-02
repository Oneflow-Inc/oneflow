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
from oneflow.nn.module import Module


class MovingAverageMinMaxObserver(Module):
    def __init__(
        self,
        training: bool = False,
        quantization_formula: str = "google",
        stop_update_after_iters: int = 0,
        quantization_bit: int = 8,
        quantization_scheme: str = "symmetric",
        momentum: float = 0,
    ) -> None:
        super().__init__()
        self.training = training
        self.quantization_formula = quantization_formula
        self.stop_update_after_iters = stop_update_after_iters
        self.quantization_bit = quantization_bit
        self.quantization_scheme = quantization_scheme
        self.momentum = momentum

    def forward(self, input, current_train_step, moving_max, moving_min):
        return flow.F.moving_average_min_max_observer(
            input,
            current_train_step,
            moving_max,
            moving_min,
            self.training,
            self.quantization_formula,
            self.stop_update_after_iters,
            self.quantization_bit,
            self.quantization_scheme,
            self.momentum,
        )


def moving_average_min_max_observer_op(
    input,
    current_train_step,
    moving_max,
    moving_min,
    training: bool = False,
    quantization_formula: str = "google",
    stop_update_after_iters: int = 0,
    quantization_bit: int = 8,
    quantization_scheme: str = "symmetric",
    momentum: float = 0,
):
    """Compute the quantization parameters based on the moving average of the input tensor's min and max values.

    First compute the moving\\_max and moving\\_min value of input tensor:

        if quantization_scheme == "symmetric":

        .. math::

            & moving\\_max = moving\\_max * momentum + |max(input)| * (1 - momentum)

            & moving\\_min = moving\\_max

        elif quantization_scheme == "affine":

        .. math::

            & moving\\_max = moving\\_max * momentum + max(input) * (1 - momentum)

            & moving\\_min = moving\\_min * momentum + min(input) * (1 - momentum)

    The moving average of min and max values are initialized as the first batch of input `Blob`'s min and max.

    Then compute the scale and zero_point with the following equations:

        if quantization_scheme == "symmetric":

        .. math::

            & denom = 2^{quantization\\_to\\_bit - 1} - 1

            & scale = moving\\_max / denom

            & zero\\_point = 0

        elif quantization_scheme == "affine":

        .. math::

            & denom = 2^{quantization\\_to\\_bit} - 1

            & scale = (moving\\_max - moving\\_min) / denom

            & zero\\_point = -moving\\_min / scale

    Args:
        input (oneflow.Tensor): input tensor.
        quantization_bit (int): Quantize input to uintX / intX, X can be in range [2, 8]. Defaults to 8.
        quantization_scheme (str): "symmetric" or "affine", quantize to signed / unsigned integer. Defaults to "symmetric".
        quantization_formula (str): Support "google" or "cambricon".
        momentum (float): Smoothing parameter for exponential moving average operation. Defaults to 0.95.

    Returns:
        Tuple[oneflow.Tensor, oneflow.Tensor]: The scale and zero_point of input tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> weight = (np.random.random((2, 3, 4, 5)) - 0.5).astype(np.float32)
        
        >>> input_tensor = flow.Tensor(
        ...    weight, dtype=flow.float32
        ... )

        >>> moving_max_np = np.zeros((1,))
        >>> moving_min_np = np.zeros((1,))
        >>> moving_max_tensor = flow.Tensor(moving_max_np)
        >>> moving_min_tensor = flow.Tensor(moving_min_np)
        >>> current_train_step_tensor = flow.Tensor(
        ...   np.zeros((1,)).astype(np.float32),
        ...    dtype=flow.int64,
        ... )
        
        >>> momentum = 0.95
        >>> quantization_bit = 8
        >>> quantization_scheme = "symmetric"
        >>> quantization_formula = "google"

        >>> (scale, zero_point) = flow.quantization.moving_average_min_max_observer(
        ...    input_tensor,
        ...    current_train_step_tensor,
        ...    moving_max_tensor,
        ...    moving_min_tensor,
        ...    True,
        ...    quantization_formula=quantization_formula,
        ...    stop_update_after_iters=1,
        ...    quantization_bit=quantization_bit,
        ...    quantization_scheme=quantization_scheme,
        ...    momentum=momentum,
        ... )

    """
    return MovingAverageMinMaxObserver(
        training=training,
        quantization_formula=quantization_formula,
        stop_update_after_iters=stop_update_after_iters,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        momentum=momentum,
    )(input, current_train_step, moving_max, moving_min)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=False)
