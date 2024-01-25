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
from typing import Optional, Tuple, Union

import oneflow as flow


def upsample(
    input,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
):
    r"""    
    Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    See :class:`~oneflow.nn.Upsample`, :class:`~oneflow.nn.UpsamplingNearest2d`,
    :class:`~oneflow.nn.UpsamplingBilinear2d` for details.
    """
    return flow.nn.functional.interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )
