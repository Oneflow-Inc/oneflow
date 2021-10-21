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
import warnings
from typing import Optional, Tuple, List

from oneflow.framework.tensor import Tensor
import oneflow as flow


def _is_tensor_a_flow_image(x: Tensor) -> bool:
    return x.ndim >= 2


def _assert_image_tensor(img):
    if not _is_tensor_a_flow_image(img):
        raise TypeError("Tensor is not a flow image.")


def _get_image_size(img: Tensor) -> List[int]:
    # Returns (w, h) of tensor image
    _assert_image_tensor(img)
    return [img.shape[-1], img.shape[-2]]


def _get_image_num_channels(img: Tensor) -> int:
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]

    raise TypeError("Input ndim should be 2 or more. Got {}".format(img.ndim))


def _max_value(dtype: flow.dtype) -> float:

    a = flow.tensor(2, dtype=dtype)
    # TODO:Tensor.is_signed()
    # signed = 1 if flow.tensor(0, dtype=dtype).is_signed() else 0
    signed = 1
    bits = 1
    max_value = flow.tensor(-signed, dtype=flow.long)
    while True:
        next_value = a.pow(bits - signed).sub(1)
        if next_value > max_value:
            max_value = next_value
            bits *= 2
        else:
            break
    return max_value.item()


def _cast_squeeze_in(
    img: Tensor, req_dtypes: List[flow.dtype]
) -> Tuple[Tensor, bool, bool, flow.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = flow._C.cast(img, req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(
    img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: flow.dtype
):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (flow.uint8, flow.int8, flow.int16, flow.int32, flow.int64):
            # it is better to round before cast
            img = flow.round(img)
        img = flow._C.cast(img, out_dtype)
    return img


def convert_image_dtype(
    image: flow.Tensor, dtype: flow.dtype = flow.float
) -> flow.Tensor:
    if image.dtype == dtype:
        return image

    if image.is_floating_point():
        # TODO:Tensor.is_floating_point()
        if flow.tensor(0, dtype=dtype).is_floating_point():
            return flow._C.cast(image, dtype)

        # float to int
        if (image.dtype == flow.float32 and dtype in (flow.int32, flow.int64)) or (
            image.dtype == flow.float64 and dtype == flow.int64
        ):
            msg = f"The cast from {image.dtype} to {dtype} cannot be performed safely."
            raise RuntimeError(msg)

        # https://github.com/pytorch/vision/pull/2078#issuecomment-612045321
        # For data in the range 0-1, (float * 255).to(uint) is only 255
        # when float is exactly 1.0.
        # `max + 1 - epsilon` provides more evenly distributed mapping of
        # ranges of floats to ints.
        eps = 1e-3
        max_val = _max_value(dtype)
        result = image.mul(max_val + 1.0 - eps)
        return flow._C.cast(result, dtype)
    else:
        input_max = _max_value(image.dtype)

        # int to float
        if flow.tensor(0, dtype=dtype).is_floating_point():
            image = flow._C.cast(image, dtype)
            return image / input_max

        output_max = _max_value(dtype)

        # int to int
        if input_max > output_max:
            factor = int((input_max + 1) // (output_max + 1))
            image = flow.div(image, factor, rounding_mode="floor")
            return flow._C.cast(image, dtype)
        else:
            factor = int((output_max + 1) // (input_max + 1))
            image = flow._C.cast(image, dtype)
            return image * factor


def vflip(img: Tensor) -> Tensor:
    _assert_image_tensor(img)

    return img.flip(-2)


def hflip(img: Tensor) -> Tensor:
    _assert_image_tensor(img)

    return img.flip(-1)


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    _assert_image_tensor(img)

    w, h = _get_image_size(img)
    right = left + width
    bottom = top + height

    if left < 0 or top < 0 or right > w or bottom > h:
        padding_ltrb = [
            max(-left, 0),
            max(-top, 0),
            max(right - w, 0),
            max(bottom - h, 0),
        ]
        return pad(
            img[..., max(top, 0) : bottom, max(left, 0) : right], padding_ltrb, fill=0
        )
    return img[..., top:bottom, left:right]


def _pad_symmetric(img: Tensor, padding: List[int]) -> Tensor:
    # padding is left, right, top, bottom

    # crop if needed
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or padding[3] < 0:
        crop_left, crop_right, crop_top, crop_bottom = [-min(x, 0) for x in padding]
        img = img[
            ...,
            crop_top : img.shape[-2] - crop_bottom,
            crop_left : img.shape[-1] - crop_right,
        ]
        padding = [max(x, 0) for x in padding]

    in_sizes = img.size()

    x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [i for i in range(padding[0] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
    right_indices = [-(i + 1) for i in range(padding[1])]  # e.g. [-1, -2, -3]
    x_indices = flow.tensor(left_indices + x_indices + right_indices, device=img.device)

    y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[2] - 1, -1, -1)]
    bottom_indices = [-(i + 1) for i in range(padding[3])]
    y_indices = flow.tensor(top_indices + y_indices + bottom_indices, device=img.device)

    ndim = img.ndim
    if ndim == 3:
        return img[:, y_indices[:, None], x_indices[None, :]]
    elif ndim == 4:
        return img[:, :, y_indices[:, None], x_indices[None, :]]
    else:
        raise RuntimeError("Symmetric padding of N-D tensors are not supported yet")


def pad(
    img: Tensor, padding: List[int], fill: int = 0, padding_mode: str = "constant"
) -> Tensor:
    _assert_image_tensor(img)

    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (int, float)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list) and len(padding) not in [1, 2, 4]:
        raise ValueError(
            "Padding must be an int or a 1, 2, or 4 element tuple, not a "
            + "{} element tuple".format(len(padding))
        )

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError(
            "Padding mode should be either constant, edge, reflect or symmetric"
        )

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 1:
        pad_left = pad_right = pad_top = pad_bottom = padding[0]
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    p = [pad_left, pad_right, pad_top, pad_bottom]

    if padding_mode == "edge":
        # remap padding_mode str
        padding_mode = "replicate"
    elif padding_mode == "symmetric":
        # route to another implementation
        return _pad_symmetric(img, p)

    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if (padding_mode != "constant") and img.dtype not in (flow.float32, flow.float64):
        # Here we temporary cast input tensor to float
        # until pytorch issue is resolved :
        # https://github.com/pytorch/pytorch/issues/40763
        need_cast = True
        img = flow._C.cast(img, flow.float32)
    img = flow._C.pad(img, pad=p, mode=padding_mode, value=float(fill))

    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        img = flow._C.cast(img, out_dtype)
    return img


def resize(img: Tensor, size: List[int], interpolation: str = "bilinear") -> Tensor:
    _assert_image_tensor(img)

    if not isinstance(size, (int, tuple, list)):
        raise TypeError("Got inappropriate size arg")
    if not isinstance(interpolation, str):
        raise TypeError("Got inappropriate interpolation arg")

    if interpolation not in ["nearest", "bilinear", "bicubic"]:
        raise ValueError("This interpolation mode is unsupported with Tensor input")

    if isinstance(size, tuple):
        size = list(size)

    if isinstance(size, list) and len(size) not in [1, 2]:
        raise ValueError(
            "Size must be an int or a 1 or 2 element tuple/list, not a "
            "{} element tuple/list".format(len(size))
        )

    w, h = _get_image_size(img)

    if isinstance(size, int):
        size_w, size_h = size, size
    elif len(size) < 2:
        size_w, size_h = size[0], size[0]
    else:
        size_w, size_h = size[1], size[0]  # Convention (h, w)

    if isinstance(size, int) or len(size) < 2:
        if w < h:
            size_h = int(size_w * h / w)
        else:
            size_w = int(size_h * w / h)

        if (w <= h and w == size_w) or (h <= w and h == size_h):
            return img

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(
        img, [flow.float32, flow.float64]
    )

    # Define align_corners to avoid warnings
    align_corners = False if interpolation in ["bilinear", "bicubic"] else None

    img = flow.nn.functional.interpolate(
        img, size=[size_h, size_w], mode=interpolation, align_corners=align_corners
    )

    if interpolation == "bicubic" and out_dtype == flow.uint8:
        img = img.clamp(min=0, max=255)

    img = _cast_squeeze_out(
        img, need_cast=need_cast, need_squeeze=need_squeeze, out_dtype=out_dtype
    )

    return img


def _assert_grid_transform_inputs(
    img: Tensor,
    matrix: Optional[List[float]],
    interpolation: str,
    fill: Optional[List[float]],
    supported_interpolation_modes: List[str],
    coeffs: Optional[List[float]] = None,
):

    if not (isinstance(img, flow.Tensor)):
        raise TypeError("Input img should be Tensor")

    _assert_image_tensor(img)

    if matrix is not None and not isinstance(matrix, list):
        raise TypeError("Argument matrix should be a list")

    if matrix is not None and len(matrix) != 6:
        raise ValueError("Argument matrix should have 6 float values")

    if coeffs is not None and len(coeffs) != 8:
        raise ValueError("Argument coeffs should have 8 float values")

    if fill is not None and not isinstance(fill, (int, float, tuple, list)):
        warnings.warn("Argument fill should be either int, float, tuple or list")

    # Check fill
    num_channels = _get_image_num_channels(img)
    if isinstance(fill, (tuple, list)) and (
        len(fill) > 1 and len(fill) != num_channels
    ):
        msg = (
            "The number of elements in 'fill' cannot broadcast to match the number of "
            "channels of the image ({} != {})"
        )
        raise ValueError(msg.format(len(fill), num_channels))

    if interpolation not in supported_interpolation_modes:
        raise ValueError(
            "Interpolation mode '{}' is unsupported with Tensor input".format(
                interpolation
            )
        )
