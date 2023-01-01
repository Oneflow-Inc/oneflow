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
import numpy as np
import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.modules.module import Module


class MovingAverageMinMaxObserver(Module):
    """
    
    Compute the quantization parameters based on the moving average of the input tensor's min and max values.

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

    Note:
        ``current_train_step`` can be directly assigned to an optimizer(eg.SGD) step.

    Args:
        input(oneflow.Tensor):  the input value(s), in ``oneflow.float32``.
        current_train_step_tensor(oneflow.Tensor): record train step for quantionzation aware training.
        stop_update_after_iters(int): stop record train step for quantionzation aware training when train iter greater than stop_update_after_iters.
        quantization_formula (str): Support "google" or "cambricon".
        quantization_bit (int): Quantize input to uintX / intX, X can be in range [2, 8]. Defaults to 8.
        quantization_scheme (str): "symmetric" or "affine", quantize to signed / unsigned integer. Defaults to "symmetric".
        momentum (float): Smoothing parameter for exponential moving average operation. Defaults to 0.95.

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

        >>> current_train_step_tensor = flow.tensor(
        ...   np.zeros((1,)).astype(np.float32),
        ...    dtype=flow.int64,
        ... )
        
        >>> momentum = 0.95
        >>> quantization_bit = 8
        >>> quantization_scheme = "symmetric"
        >>> quantization_formula = "google"

        >>> moving_average_min_max_observer = flow.nn.MovingAverageMinMaxObserver(stop_update_after_iters=1,  
        ...                                                                       quantization_formula=quantization_formula, quantization_bit=quantization_bit,
        ...                                                                       quantization_scheme=quantization_scheme, momentum=momentum,
        ...                                                                       )

        >>> (scale, zero_point) = moving_average_min_max_observer(
        ...    input_tensor,
        ...    current_train_step_tensor,
        ... )

    """

    def __init__(
        self,
        stop_update_after_iters: int = 1,
        quantization_formula: str = "google",
        quantization_bit: int = 8,
        quantization_scheme: str = "symmetric",
        momentum: float = 0.95,
    ) -> None:
        super().__init__()
        self.quantization_formula = quantization_formula
        self.stop_update_after_iters = stop_update_after_iters
        self.quantization_bit = quantization_bit
        self.quantization_scheme = quantization_scheme
        self.momentum = momentum
        self.register_buffer("moving_max", flow.Tensor(1))
        self.register_buffer("moving_min", flow.Tensor(1))
        self.reset_running_stats()

    def reset_running_stats(self) -> None:
        self.moving_max.fill_(0)
        self.moving_min.fill_(0)

    def forward(self, input, current_train_step):
        return flow._C.moving_average_min_max_observer(
            input,
            current_train_step,
            self.moving_max,
            self.moving_min,
            self.training,
            self.stop_update_after_iters,
            self.quantization_formula,
            self.quantization_bit,
            self.quantization_scheme,
            self.momentum,
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
