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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from typing import Optional, Union, Tuple


@oneflow_export("nn.Upsample")
@experimental_api
class Upsample(Module):
    r"""
    
    """

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None, 
        scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
        mode: str = 'nearest', 
        align_corners: Optional[bool] = None
    ):
        super().__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        
        self.height_scale = None
        self.width_scale = None

        if isinstance(self.scale_factor, float):
            self.height_scale = self.scale_factor
            self.width_scale = self.scale_factor
        elif isinstance(self.scale_factor, tuple):
            self.height_scale = self.scale_factor[0]
            self.width_scale = self.scale_factor[1]
        else:
            pass
        
        if self.mode != "nearest" and self.mode != "bilinear":
            raise ValueError('interpolation must be "nearest" or "bilinear".')

        if self.mode == "nearest" and self.align_corners:
            raise ValueError('interpolation "nearest" does not support align_corners.')
        
        self.op = (
            flow.builtin_op("upsample")
                .Input("x")
                .Output("y")
                .Attr("height_scale")
                .Attr("width_scale")
                .Attr("align_corners", align_corners)
                .Attr("data_format", "channels_first")
                .Attr("interpolation", mode)
                .Build()
        )

    def forward(self, x):
        assert self.size != None or self.scale_factor != None, f"size and scale_factor can not be none at the same time!"
        h, w = x.shape[2], x.shape[3]
        if self.height_scale == None:
            if isinstance(self.size, int):
                self.height_scale = self.size / h
            else:
                self.height_scale = self.size[0] / h
        if self.width_scale == None:
            if isinstance(self.size, int):
                self.width_scale = self.size / w
            else:
                self.width_scale = self.size[1] / w

        res = self._op(x, height_scale=height_scale, width_scale=width_scale)[0]
        return res
