import math
import numbers
import warnings
from enum import Enum

import numpy as np
from PIL import Image
from typing import List, Tuple, Any, Optional

try:
    import accimage
except ImportError:
    accimage = None

import oneflow.experimental as flow
from oneflow.experimental import Tensor

from . import functional_pil as F_pil
from . import functional_tensor as F_t

class InterpolationMode(Enum):
    """Interpolation modes
    """
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"

def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]

pil_modes_mapping = {
    InterpolationMode.NEAREST: 0,
    InterpolationMode.BILINEAR: 2,
    InterpolationMode.BICUBIC: 3,
    InterpolationMode.BOX: 4,
    InterpolationMode.HAMMING: 5,
    InterpolationMode.LANCZOS: 1,
}

def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    This function does not support torchscript.
    See :class:`~torchvision.transforms.ToTensor` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

    # default_float_dtype = flow.get_default_dtype()
    default_float_dtype = flow.float32

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = flow.Tensor(pic.transpose((2, 0, 1)))
        # backward compatibility
        if img.dtype == flow.int:
            return img.to(dtype=default_float_dtype).div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return flow.Tensor(nppic).to(dtype=default_float_dtype)

    # handle PIL Image
    mode_to_nptype = {'I': np.int32, 'I;16': np.int16, 'F': np.float32}
    if mode_to_nptype.get(pic.mode, np.uint8) == np.uint8:
        dtype = flow.int32
    else:
        dtype = flow.float32

    img = flow.Tensor(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True), dtype=dtype,
    )
    flow.sub

    if pic.mode == '1':
        img = 255 * img

    img = img.reshape(shape=(pic.size[1], pic.size[0], len(pic.getbands())))
    # put it from HWC to CHW format
    res = img.permute(2, 0, 1)
    if img.dtype == flow.int:
        res = res.to(dtype=default_float_dtype).div(255)
    return res 


def resize(img: Tensor, size: List[int], interpolation: InterpolationMode = InterpolationMode.BILINEAR) -> Tensor:
    r"""Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        img (PIL Image or Tensor): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.
            In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    Returns:
        PIL Image or Tensor: Resized image.
    """
    # Backward compatibility with integer value
    if isinstance(interpolation, int):
        warnings.warn(
            "Argument interpolation should be of type InterpolationMode instead of int. "
            "Please, use InterpolationMode enum."
        )
        interpolation = _interpolation_modes_from_int(interpolation)

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if not isinstance(img, flow.Tensor):
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil.resize(img, size=size, interpolation=pil_interpolation)

    return F_t.resize(img, size=size, interpolation=interpolation.value)
