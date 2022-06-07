import oneflow as flow
import oneflow.nn as nn
from oneflow.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional

def get_fake_quantized(
    input, input_observer, current_train_step, fake_quantizer
):
    in_scale, in_zero_point = input_observer(input, current_train_step)
    input_fake_quanted = fake_quantizer(input, in_scale, in_zero_point)
    return input_fake_quanted

def init_fake_quants(
    self,
    quantization_formula: str = "google",
    quantization_bit: int = 8,
    quantization_scheme: str = "symmetric",
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
    self.fake_quantizer = nn.FakeQuantization(
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
    )

class QuantMaxPool1d(nn.MaxPool1d):
    r"""Quantized 1D max pool.
    A MaxPool1d module attached with MinMaxObserver, MovingAverageMinMaxObserver and FakeQuantize modules for input,
    used for quantization aware training.
    The parameters of MaxPool1d are the same as :class:`~oneflow.nn.MaxPool1d` with some extra parameters for fake quantization.
    """
    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: Optional[_size_1_t] = None,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        quantization_formula: str = "google",
        quantization_bit: int = 8,
        quantization_scheme: str = "symmetric",
        input_quant_momentum: float = 0.95,
    ):
        super().__init__(
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            ceil_mode
        )
        init_fake_quants(
            self,
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            input_quant_momentum=input_quant_momentum,
        )

    
    def forward(self, x):
        fake_quan_input = get_fake_quantized(
            x,
            self.input_min_max_observer,
            self.current_train_step,
            self.fake_quantizer,
        )
        return super().forward(fake_quan_input)


class QuantMaxPool2d(nn.MaxPool2d):
    r"""Quantized 2D max pool.
    A MaxPool2d module attached with MinMaxObserver, MovingAverageMinMaxObserver and FakeQuantize modules for input,
    used for quantization aware training.
    The parameters of MaxPool2d are the same as :class:`~oneflow.nn.MaxPool2d` with some extra parameters for fake quantization.
    """
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        quantization_formula: str = "google",
        quantization_bit: int = 8,
        quantization_scheme: str = "symmetric",
        input_quant_momentum: float = 0.95,
    ):
        super().__init__(
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            ceil_mode
        )
        init_fake_quants(
            self,
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            input_quant_momentum=input_quant_momentum,
        )

    
    def forward(self, x):
        fake_quan_input = get_fake_quantized(
            x,
            self.input_min_max_observer,
            self.current_train_step,
            self.fake_quantizer,
        )
        return super().forward(fake_quan_input)


class QuantMaxPool3d(nn.MaxPool3d):
    r"""Quantized 3D max pool.
    A MaxPool3d module attached with MinMaxObserver, MovingAverageMinMaxObserver and FakeQuantize modules for input,
    used for quantization aware training.
    The parameters of MaxPool3d are the same as :class:`~oneflow.nn.MaxPool3d` with some extra parameters for fake quantization.
    """
    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        quantization_formula: str = "google",
        quantization_bit: int = 8,
        quantization_scheme: str = "symmetric",
        input_quant_momentum: float = 0.95,
    ):
        super().__init__(
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            ceil_mode
        )
        init_fake_quants(
            self,
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            input_quant_momentum=input_quant_momentum,
        )

    
    def forward(self, x):
        fake_quan_input = get_fake_quantized(
            x,
            self.input_min_max_observer,
            self.current_train_step,
            self.fake_quantizer,
        )
        return super().forward(fake_quan_input)
