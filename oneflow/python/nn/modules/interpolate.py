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
import math
import warnings
import oneflow as flow
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from typing import Optional, Union, Tuple


class INTERPOLATE(Module):
    r"""
    """

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
    ):
        super().__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None

        if mode in ("nearest", "area") and align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the "
                "interpolating modes: linear | bilinear | bicubic | trilinear"
            )

        self.mode = mode
        self.recompute_scale_factor = recompute_scale_factor
        if align_corners == None:
            align_corners = False

        self.align_corners = align_corners
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

        if (
            self.mode != "nearest"
            and self.mode != "bilinear"
            and self.mode != "linear"
            and self.mode != "area"
            and self.mode != "bicubic"
            and self.mode != "trilinear"
        ):
            raise ValueError(
                'interpolation must be "nearest" or "bilinear" or "linear" or "area" or "bicubic" or "trilinear".'
            )

        if self.mode == "nearest" and self.align_corners:
            raise ValueError('interpolation "nearest" does not support align_corners.')

    def forward(self, x):
        dim = len(x.shape) - 2
        if self.size is not None and self.scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        elif self.size is not None:
            assert self.scale_factor is None
            scale_factors = None
            if isinstance(self.size, (list, tuple)):
                if len(self.size) != dim:
                    raise ValueError(
                        "size shape must match input shape. "
                        "Input is {}D, size is {}".format(dim, len(self.size))
                    )
                output_size = self.size
            else:
                output_size = [self.size for _ in range(dim)]
        elif self.scale_factor is not None:
            assert self.size is None
            output_size = None
            if isinstance(self.scale_factor, (list, tuple)):
                if len(self.scale_factor) != dim:
                    raise ValueError(
                        "scale_factor shape must match input shape. "
                        "Input is {}D, scale_factor is {}".format(
                            dim, len(self.scale_factor)
                        )
                    )
                scale_factors = self.scale_factor
            else:
                scale_factors = [self.scale_factor for _ in range(dim)]
        else:
            raise ValueError("either size or scale_factor should be defined")

        if self.recompute_scale_factor is None:
            if scale_factors is not None:
                for scale in scale_factors:
                    if math.floor(scale) != scale:
                        warnings.warn(
                        "The default behavior for interpolate/upsample with float scale_factor changed "
                        "in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, "
                        "instead of relying on the computed output size. "
                        "If you wish to restore the old behavior, please set recompute_scale_factor=True. "
                        "See the documentation of nn.Upsample for details. "
                    )
                    break
        elif self.recompute_scale_factor and self.size is not None:
            raise ValueError("recompute_scale_factor is not meaningful with an explicit size.")

        # "area" mode always requires an explicit size rather than scale factor.
        # Re-use the recompute_scale_factor code path.
        if self.mode == "area" and output_size is None:
            self.recompute_scale_factor = True
        
        if self.recompute_scale_factor is not None and self.recompute_scale_factor:
            assert scale_factors is not None
            output_size = [int(math.floor(float(input.size(i + 2)) * scale_factors[i])) for i in range(dim)]
            scale_factors = None

        if len(x.shape) == 3 and self.mode == "nearest":
            return flow.F.upsample_nearest_1d(
                x, scale_factor=scale_factors[0], data_format="channels_first"
            )

        if len(x.shape) == 4 and self.mode == "nearest":
            return flow.F.upsample_nearest_2d(
                x,
                height_scale=scale_factors[0],
                width_scale=scale_factors[1],
                data_format="channels_first",
            )

        if len(x.shape) == 5 and self.mode == "nearest":
            return flow.F.upsample_nearest_3d(
                x,
                depth_scale=scale_factors[0],
                height_scale=scale_factors[1],
                width_scale=scale_factors[2],
                data_format="channels_first",
            )

        # TODO(bbuf) Add adaptive_avg_pool op

        if self.mode == "area":
            raise NotImplementedError("adaptive_avg_pool1d not impleted now!")

        if len(x.shape) == 3 and self.mode == "linear":
            assert self.align_corners is not None
            return flow.F.upsample_linear_1d(
                x,
                scale_factor=scale_factors[0],
                align_corners=self.align_corners,
                data_format="channels_first",
            )

        if len(x.shape) == 4 and self.mode == "bilinear":
            assert self.align_corners is not None
            return flow.F.upsample_bilinear_2d(
                x,
                height_scale=scale_factors[0],
                width_scale=scale_factors[1],
                align_corners=self.align_corners,
                data_format="channels_first",
            )

        if len(x.shape) == 4 and self.mode == "bicubic":
            assert self.align_corners is not None
            return flow.F.upsample_bicubic_2d(
                x,
                height_scale=scale_factors[0],
                width_scale=scale_factors[1],
                align_corners=self.align_corners,
                data_format="channels_first",
            )

        if len(x.shape) == 5 and self.mode == "trilinear":
            assert self.align_corners is not None
            return flow.F.upsample_trilinear_3d(
                x,
                depth_scale=scale_factors[0],
                height_scale=scale_factors[1],
                width_scale=scale_factors[2],
                align_corners=self.align_corners,
                data_format="channels_first",
            )


@oneflow_export("nn.functional.interpolate")
@experimental_api
def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None
):
    return INTERPOLATE(
        size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor
    )(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
