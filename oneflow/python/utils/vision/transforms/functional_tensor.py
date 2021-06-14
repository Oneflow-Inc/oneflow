import warnings

import oneflow.experimental as flow
from oneflow.experimental import Tensor
# from oneflow.experimental.nn.functional import grid_sample, conv2d, interpolate, pad as flow_pad
from typing import Optional, Tuple, List


def _is_tensor_a_flow_image(x: Tensor) -> bool:
    return x.ndim >= 2


def _assert_image_tensor(img):
    if not _is_tensor_a_flow_image(img):
        raise TypeError("Tensor is not a torch image.")


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


def _cast_squeeze_in(img: Tensor, req_dtypes: List[flow.dtype]) -> Tuple[Tensor, bool, bool, flow.dtype]:
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
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: flow.dtype):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (flow.uint8, flow.int8, flow.int16, flow.int32, flow.int64):
            # it is better to round before cast
            img = flow.round(img)
        img = img.to(out_dtype)

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
        raise ValueError("Size must be an int or a 1 or 2 element tuple/list, not a "
                         "{} element tuple/list".format(len(size)))

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

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [flow.float32, flow.float64])

    # Define align_corners to avoid warnings
    align_corners = False if interpolation in ["bilinear", "bicubic"] else None

    # TODO: nn.functional.interpolate
    raise NotImplementedError("Not support for now because nn.functional.interpolate is not ready!")
    # img = interpolate(img, size=[size_h, size_w], mode=interpolation, align_corners=align_corners)

    if interpolation == "bicubic" and out_dtype == flow.uint8:
        img = img.clamp(min=0, max=255)

    img = _cast_squeeze_out(img, need_cast=need_cast, need_squeeze=need_squeeze, out_dtype=out_dtype)

    return img

