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
import numbers
from typing import Any, List, Sequence

import numpy as np
import oneflow.experimental as flow
from PIL import (
    Image,
    ImageOps,
    ImageEnhance,
    ImageFilter,
    __version__ as PILLOW_VERSION,
)

try:
    import accimage
except ImportError:
    accimage = None


def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def resize(img, size, interpolation=Image.BILINEAR):
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))
    if not (
        isinstance(size, int) or (isinstance(size, Sequence) and len(size) in (1, 2))
    ):
        raise TypeError("Got inappropriate size arg: {}".format(size))

    if isinstance(size, int) or len(size) == 1:
        if isinstance(size, Sequence):
            size = size[0]
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)
