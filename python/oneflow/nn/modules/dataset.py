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
import os
import random
import sys
import traceback
from typing import List, Optional, Sequence, Tuple, Union

import oneflow as flow
import oneflow._oneflow_internal._C as _C
from oneflow.framework.tensor import Tensor
from oneflow.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _pair, _reverse_repeat_tuple, _single, _triple
import oneflow.framework.id_util as id_util


def mirrored_gen_random_seed(seed=None):
    if seed is None:
        seed = -1
        has_seed = False
    else:
        has_seed = True
    return (seed, has_seed)


class OFRecordReader(Module):
    def __init__(
        self,
        ofrecord_dir: str,
        batch_size: int = 1,
        data_part_num: int = 1,
        part_name_prefix: str = "part-",
        part_name_suffix_length: int = -1,
        random_shuffle: bool = False,
        shuffle_buffer_size: int = 1024,
        shuffle_after_epoch: bool = False,
        random_seed: int = -1,
        device: Union[flow.device, str] = None,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
        name: Optional[str] = None,
    ):
        super().__init__()

        if name is not None:
            print("WARNING: name has been deprecated and has NO effect.\n")
        self.ofrecord_dir = ofrecord_dir
        self.batch_size = batch_size
        self.data_part_num = data_part_num
        self.part_name_prefix = part_name_prefix
        self.part_name_suffix_length = part_name_suffix_length
        self.random_shuffle = random_shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_after_epoch = shuffle_after_epoch

        self.placement = placement
        if placement is None:
            self.device = device or flow.device("cpu")
        else:
            assert device is None

        if placement is not None:
            assert isinstance(sbp, (flow.sbp.sbp, tuple, list)), "sbp: %s" % sbp
            if isinstance(sbp, flow.sbp.sbp):
                sbp = (sbp,)
            else:
                for elem in sbp:
                    assert isinstance(elem, flow.sbp.sbp), "sbp: %s" % sbp
            assert len(sbp) == len(placement.hierarchy)
        else:
            assert sbp is None, "sbp: %s" % sbp

        self.sbp = sbp

        (self.seed, self.has_seed) = mirrored_gen_random_seed(random_seed)
        self._op = flow.stateful_op("OFRecordReader").Output("out").Build()

    def forward(self):
        if self.placement is not None:
            res = _C.dispatch_ofrecord_reader(
                self._op,
                data_dir=self.ofrecord_dir,
                data_part_num=self.data_part_num,
                part_name_prefix=self.part_name_prefix,
                part_name_suffix_length=self.part_name_suffix_length,
                batch_size=self.batch_size,
                shuffle_buffer_size=self.shuffle_buffer_size,
                random_shuffle=self.random_shuffle,
                shuffle_after_epoch=self.shuffle_after_epoch,
                seed=self.seed,
                sbp=self.sbp,
                placement=self.placement,
            )
        else:
            res = _C.dispatch_ofrecord_reader(
                self._op,
                data_dir=self.ofrecord_dir,
                data_part_num=self.data_part_num,
                part_name_prefix=self.part_name_prefix,
                part_name_suffix_length=self.part_name_suffix_length,
                batch_size=self.batch_size,
                shuffle_buffer_size=self.shuffle_buffer_size,
                random_shuffle=self.random_shuffle,
                shuffle_after_epoch=self.shuffle_after_epoch,
                seed=self.seed,
                device=self.device,
            )
        return res


class OFRecordRawDecoder(Module):
    def __init__(
        self,
        blob_name: str,
        shape: Sequence[int],
        dtype: flow.dtype,
        dim1_varying_length: bool = False,
        truncate: bool = False,
        auto_zero_padding: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()
        if auto_zero_padding:
            print(
                "WARNING: auto_zero_padding has been deprecated, Please use truncate instead.\n"
            )
        if name is not None:
            print("WARNING: name has been deprecated and has NO effect.\n")
        self.blob_name = blob_name
        self.shape = shape
        self.dtype = dtype
        self.dim1_varying_length = dim1_varying_length
        self.truncate = truncate
        self.auto_zero_padding = auto_zero_padding
        self._op = (
            flow.stateful_op("ofrecord_raw_decoder").Input("in").Output("out").Build()
        )

    def forward(self, input):
        res = _C.dispatch_ofrecord_raw_decoder(
            self._op,
            input,
            name=self.blob_name,
            shape=self.shape,
            data_type=self.dtype,
            dim1_varying_length=self.dim1_varying_length,
            truncate=self.truncate or self.auto_zero_padding,
        )
        return res


class CoinFlip(Module):
    def __init__(
        self,
        batch_size: int = 1,
        random_seed: Optional[int] = None,
        probability: float = 0.5,
        device: Union[flow.device, str] = None,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.probability = probability

        self.placement = placement
        if placement is None:
            if device is None:
                self.device = flow.device("cpu")
        else:
            assert device is None

        if placement is not None:
            assert isinstance(sbp, (flow.sbp.sbp, tuple, list)), "sbp: %s" % sbp
            if isinstance(sbp, flow.sbp.sbp):
                sbp = (sbp,)
            else:
                for elem in sbp:
                    assert isinstance(elem, flow.sbp.sbp), "sbp: %s" % sbp
            assert len(sbp) == len(placement.hierarchy)
        else:
            assert sbp is None, "sbp: %s" % sbp

        self.sbp = sbp

        (self.seed, self.has_seed) = mirrored_gen_random_seed(random_seed)

        self._op = flow.stateful_op("coin_flip").Output("out").Build()

    def forward(self):
        if self.placement is not None:
            res = _C.dispatch_coin_flip(
                self._op,
                batch_size=self.batch_size,
                probability=self.probability,
                has_seed=self.has_seed,
                seed=self.seed,
                placement=self.placement,
                sbp=self.sbp,
            )
        else:
            res = _C.dispatch_coin_flip(
                self._op,
                batch_size=self.batch_size,
                probability=self.probability,
                has_seed=self.has_seed,
                seed=self.seed,
                device=self.device,
            )
        return res


class CropMirrorNormalize(Module):
    def __init__(
        self,
        color_space: str = "BGR",
        output_layout: str = "NCHW",
        crop_h: int = 0,
        crop_w: int = 0,
        crop_pos_y: float = 0.5,
        crop_pos_x: float = 0.5,
        mean: Sequence[float] = [0.0],
        std: Sequence[float] = [1.0],
        output_dtype: flow.dtype = flow.float,
    ):
        super().__init__()
        if output_layout != "NCHW":
            print(
                "WARNING: output_layout has been deprecated. Please use Environment Variable ONEFLOW_ENABLE_NHWC, and make it equals 1."
            )
        if os.getenv("ONEFLOW_ENABLE_NHWC") == "1":
            output_layout = "NHWC"
        else:
            output_layout = "NCHW"

        self.color_space = color_space
        self.output_layout = output_layout
        self.mean = mean
        self.std = std
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.crop_pos_y = crop_pos_y
        self.crop_pos_x = crop_pos_x
        self.output_dtype = output_dtype

        self._op_uint8_with_mirror = (
            flow.stateful_op("crop_mirror_normalize_from_uint8")
            .Input("in")
            .Input("mirror")
            .Output("out")
            .Build()
        )
        self._op_uint8_no_mirror = (
            flow.stateful_op("crop_mirror_normalize_from_uint8")
            .Input("in")
            .Output("out")
            .Build()
        )
        self._op_buffer_with_mirror = (
            flow.stateful_op("crop_mirror_normalize_from_tensorbuffer")
            .Input("in")
            .Input("mirror")
            .Output("out")
            .Build()
        )

        self._op_buffer_no_mirror = (
            flow.stateful_op("crop_mirror_normalize_from_tensorbuffer")
            .Input("in")
            .Output("out")
            .Build()
        )

    def forward(self, input, mirror=None):
        if input.dtype is flow.uint8:
            if mirror is not None:
                res = _C.dispatch_crop_mirror_normalize_from_uint8(
                    self._op_uint8_with_mirror,
                    (input, mirror),
                    color_space=self.color_space,
                    output_layout=self.output_layout,
                    mean=self.mean,
                    std=self.std,
                    crop_h=self.crop_h,
                    crop_w=self.crop_w,
                    crop_pos_x=self.crop_pos_x,
                    crop_pos_y=self.crop_pos_y,
                    output_dtype=self.output_dtype,
                )
            else:
                res = _C.dispatch_crop_mirror_normalize_from_uint8(
                    self._op_uint8_no_mirror,
                    (input,),
                    color_space=self.color_space,
                    output_layout=self.output_layout,
                    mean=self.mean,
                    std=self.std,
                    crop_h=self.crop_h,
                    crop_w=self.crop_w,
                    crop_pos_x=self.crop_pos_x,
                    crop_pos_y=self.crop_pos_y,
                    output_dtype=self.output_dtype,
                )
        elif input.dtype is flow.tensor_buffer:
            if mirror is not None:
                res = _C.dispatch_crop_mirror_normalize_from_tensorbuffer(
                    self._op_buffer_with_mirror,
                    (input, mirror),
                    color_space=self.color_space,
                    output_layout=self.output_layout,
                    mean=self.mean,
                    std=self.std,
                    crop_h=self.crop_h,
                    crop_w=self.crop_w,
                    crop_pos_x=self.crop_pos_x,
                    crop_pos_y=self.crop_pos_y,
                    output_dtype=self.output_dtype,
                )
            else:
                res = _C.dispatch_crop_mirror_normalize_from_tensorbuffer(
                    self._op_buffer_no_mirror,
                    (input,),
                    color_space=self.color_space,
                    output_layout=self.output_layout,
                    mean=self.mean,
                    std=self.std,
                    crop_h=self.crop_h,
                    crop_w=self.crop_w,
                    crop_pos_x=self.crop_pos_x,
                    crop_pos_y=self.crop_pos_y,
                    output_dtype=self.output_dtype,
                )
        else:
            print(
                "ERROR! oneflow.nn.CropMirrorNormalize module NOT support input dtype = ",
                input.dtype,
            )
            raise NotImplementedError
        return res


class OFRecordImageDecoderRandomCrop(Module):
    def __init__(
        self,
        blob_name: str,
        color_space: str = "BGR",
        num_attempts: int = 10,
        random_seed: Optional[int] = None,
        random_area: Sequence[float] = [0.08, 1.0],
        random_aspect_ratio: Sequence[float] = [0.75, 1.333333],
    ):
        super().__init__()
        self.blob_name = blob_name
        self.color_space = color_space
        self.num_attempts = num_attempts
        self.random_area = random_area
        self.random_aspect_ratio = random_aspect_ratio
        (self.seed, self.has_seed) = mirrored_gen_random_seed(random_seed)
        self._op = (
            flow.stateful_op("ofrecord_image_decoder_random_crop")
            .Input("in")
            .Output("out")
            .Build()
        )

    def forward(self, input):
        res = _C.dispatch_ofrecord_image_decoder_random_crop(
            self._op,
            input,
            name=self.blob_name,
            color_space=self.color_space,
            num_attempts=self.num_attempts,
            random_area=self.random_area,
            random_aspect_ratio=self.random_aspect_ratio,
            has_seed=self.has_seed,
            seed=self.seed,
        )
        return res


class OFRecordImageDecoder(Module):
    def __init__(self, blob_name: str, color_space: str = "BGR"):
        super().__init__()
        self._op = (
            flow.stateful_op("ofrecord_image_decoder").Input("in").Output("out").Build()
        )
        self.blob_name = blob_name
        self.color_space = color_space

    def forward(self, input):
        res = _C.dispatch_ofrecord_image_decoder(
            self._op, input, name=self.blob_name, color_space=self.color_space
        )
        return res


class OFRecordImageGpuDecoderRandomCropResize(Module):
    def __init__(
        self,
        target_width: int,
        target_height: int,
        num_attempts: Optional[int] = 10,
        seed: Optional[int] = 0,
        random_area: Optional[Sequence[float]] = [0.08, 1.0],
        random_aspect_ratio: Optional[Sequence[float]] = [0.75, 1.333333],
        num_workers: Optional[int] = 3,
        warmup_size: Optional[int] = 6400,
        max_num_pixels: Optional[int] = 67108864,
    ):
        super().__init__()
        self.target_width = target_width
        self.target_height = target_height
        self.num_attempts = num_attempts
        self.seed = seed
        assert len(random_area) == 2
        self.random_area = random_area
        assert len(random_aspect_ratio) == 2
        self.random_aspect_ratio = random_aspect_ratio
        self.num_workers = num_workers
        self.warmup_size = warmup_size
        self.max_num_pixels = max_num_pixels
        gpu_decoder_conf = (
            flow._oneflow_internal.oneflow.core.operator.op_conf.ImageDecoderRandomCropResizeOpConf()
        )
        gpu_decoder_conf.set_in("error_input_need_to_be_replaced")
        gpu_decoder_conf.set_out("out")
        self._op = flow._oneflow_internal.one.ImageDecoderRandomCropResizeOpExpr(
            id_util.UniqueStr("ImageGpuDecoder"), gpu_decoder_conf, ["in"], ["out"]
        )

    def forward(self, input):
        if not input.is_lazy:
            print(
                "ERROR! oneflow.nn.OFRecordImageGpuDecoderRandomCropResize module ",
                "NOT support run as eager module, please use it in nn.Graph.",
            )
            raise NotImplementedError
        res = _C.dispatch_image_decoder_random_crop_resize(
            self._op,
            input,
            target_width=self.target_width,
            target_height=self.target_height,
            num_attempts=self.num_attempts,
            seed=self.seed,
            random_area_min=self.random_area[0],
            random_area_max=self.random_area[1],
            random_aspect_ratio_min=self.random_aspect_ratio[0],
            random_aspect_ratio_max=self.random_aspect_ratio[1],
            num_workers=self.num_workers,
            warmup_size=self.warmup_size,
            max_num_pixels=self.max_num_pixels,
        )
        if not res.is_cuda:
            print(
                "WARNING! oneflow.nn.OFRecordImageGpuDecoderRandomCropResize ONLY support ",
                "CUDA runtime version >= 10.2, so now it degenerates into CPU decode version.",
            )
        return res


class TensorBufferToListOfTensors(Module):
    def __init__(
        self, out_shapes, out_dtypes, out_num: int = 1, dynamic_out: bool = False
    ):
        super().__init__()
        self._op = (
            flow.stateful_op("tensor_buffer_to_list_of_tensors_v2")
            .Input("in")
            .Output("out", out_num)
            .Build()
        )
        self.out_shapes = out_shapes
        self.out_dtypes = out_dtypes
        self.dynamic_out = dynamic_out

    def forward(self, input):
        return _C.dispatch_tensor_buffer_to_list_of_tensors_v2(
            self._op,
            input,
            out_shapes=self.out_shapes,
            out_dtypes=self.out_dtypes,
            dynamic_out=self.dynamic_out,
        )


def tensor_buffer_to_list_of_tensors(tensor, out_shapes, out_dtypes):
    return TensorBufferToListOfTensors(
        [list(out_shape) for out_shape in out_shapes], out_dtypes, len(out_shapes)
    )(tensor)


class ImageResize(Module):
    def __init__(
        self,
        target_size: Union[int, Sequence[int]] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        keep_aspect_ratio: bool = False,
        resize_side: str = "shorter",
        channels: int = 3,
        dtype: Optional[flow.dtype] = None,
        interpolation_type: str = "auto",
        name: Optional[str] = None,
        color_space: Optional[str] = None,
        interp_type: Optional[str] = None,
        resize_shorter: int = 0,
        resize_x: int = 0,
        resize_y: int = 0,
    ):
        super().__init__()
        if name is not None:
            print("WARNING: name has been deprecated and has NO effect.\n")
        deprecated_param_used = False
        if color_space is not None:
            print(
                "WARNING: color_space has been deprecated. Please use channels instead."
            )
            print(traceback.format_stack()[-2])
            deprecated_param_used = True
            assert isinstance(color_space, str)
            if color_space.upper() == "RGB" or color_space.upper() == "BGR":
                channels = 3
            elif color_space.upper() == "GRAY":
                channels = 1
            else:
                raise ValueError("invalid color_space")
        self.channels = channels
        if interp_type is not None:
            print(
                "WARNING: interp_type has been deprecated. Please use interpolation_type instead."
            )
            print(traceback.format_stack()[-2])
            deprecated_param_used = True
            assert isinstance(interp_type, str)
            if interp_type == "Linear":
                interpolation_type = "bilinear"
            elif interp_type == "NN":
                interpolation_type = "nearest_neighbor"
            elif interp_type == "Cubic":
                interpolation_type = "bicubic"
            else:
                raise ValueError("invalid interp_type")
        self.interpolation_type = interpolation_type

        if resize_x > 0 and resize_y > 0:
            print(
                "WARNING: resize_x and resize_y has been deprecated. Please use target_size instead."
            )
            print(traceback.format_stack()[-2])
            deprecated_param_used = True
            target_size = (resize_x, resize_y)
            keep_aspect_ratio = False
        if resize_shorter > 0:
            print(
                "WARNING: resize_shorter has been deprecated. Please use target_size instead."
            )
            print(traceback.format_stack()[-2])
            deprecated_param_used = True
            target_size = resize_shorter
            keep_aspect_ratio = True
            resize_side = "shorter"
        self.keep_aspect_ratio = keep_aspect_ratio
        if self.keep_aspect_ratio:
            if not isinstance(target_size, int):
                raise ValueError(
                    "target_size must be an int when keep_aspect_ratio is True"
                )
            if min_size is None:
                min_size = 0
            if max_size is None:
                max_size = 0
            if resize_side == "shorter":
                resize_longer = False
            elif resize_side == "longer":
                resize_longer = True
            else:
                raise ValueError('resize_side must be "shorter" or "longer"')
            self.target_size = target_size
            self.min_size = min_size
            self.max_size = max_size
            self.resize_longer = resize_longer
            self._op = (
                flow.stateful_op("image_resize_keep_aspect_ratio")
                .Input("in")
                .Output("out")
                .Output("size")
                .Output("scale")
                .Build()
            )
        else:
            if (
                not isinstance(target_size, (list, tuple))
                or len(target_size) != 2
                or (not all((isinstance(size, int) for size in target_size)))
            ):
                raise ValueError(
                    "target_size must be a form like (width, height) when keep_aspect_ratio is False"
                )
            if dtype is None:
                dtype = flow.uint8
            self.dtype = dtype
            (self.target_w, self.target_h) = target_size
            self._op = (
                flow.stateful_op("image_resize_to_fixed")
                .Input("in")
                .Output("out")
                .Output("scale")
                .Build()
            )

    def forward(self, input):
        if self.keep_aspect_ratio:
            res = _C.dispatch_image_resize_keep_aspect_ratio(
                self._op,
                input,
                target_size=self.target_size,
                min_size=self.min_size,
                max_size=self.max_size,
                resize_longer=self.resize_longer,
                interpolation_type=self.interpolation_type,
            )
            new_size = flow.tensor_buffer_to_tensor(
                res[1], dtype=flow.int32, instance_shape=(2,)
            )
            scale = flow.tensor_buffer_to_tensor(
                res[2], dtype=flow.float32, instance_shape=(2,)
            )
        else:
            res = _C.dispatch_image_resize_to_fixed(
                self._op,
                input,
                target_width=self.target_w,
                target_height=self.target_h,
                channels=self.channels,
                data_type=self.dtype,
                interpolation_type=self.interpolation_type,
            )
            new_size = None
            scale = res[1]
        res_image = res[0]
        return (res_image, scale, new_size)


def raw_decoder(
    input_record,
    blob_name: str,
    shape: Sequence[int],
    dtype: flow.dtype,
    dim1_varying_length: bool = False,
    truncate: bool = False,
    auto_zero_padding: bool = False,
    name: Optional[str] = None,
):
    if auto_zero_padding:
        print(
            "WARNING: auto_zero_padding has been deprecated, Please use truncate instead.\n            "
        )
    return OFRecordRawDecoder(
        blob_name,
        shape,
        dtype,
        dim1_varying_length,
        truncate or auto_zero_padding,
        name,
    ).forward(input_record)


def get_ofrecord_handle(
    ofrecord_dir: str,
    batch_size: int = 1,
    data_part_num: int = 1,
    part_name_prefix: str = "part-",
    part_name_suffix_length: int = -1,
    random_shuffle: bool = False,
    shuffle_buffer_size: int = 1024,
    shuffle_after_epoch: bool = False,
    name: Optional[str] = None,
):
    return OFRecordReader(
        ofrecord_dir,
        batch_size,
        data_part_num,
        part_name_prefix,
        part_name_suffix_length,
        random_shuffle,
        shuffle_buffer_size,
        shuffle_after_epoch,
        name,
    )()


class ImageFlip(Module):
    """This operator flips the images.

    The flip code corresponds to the different flip mode:

    0 (0x00): Non Flip

    1 (0x01): Horizontal Flip

    2 (0x02): Vertical Flip

    3 (0x03): Both Horizontal and Vertical Flip

    Args:
        images: The input images.
        flip_code: The flip code.

    Returns:
        The result image.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> arr = np.array([
        ...    [[[1, 2, 3], [3, 2, 1]],
        ...     [[2, 3, 4], [4, 3, 2]]],
        ...    [[[3, 4, 5], [5, 4, 3]],
        ...     [[4, 5, 6], [6, 5, 4]]]])
        >>> image_tensors = flow.Tensor(arr, device=flow.device("cpu"))
        >>> image_tensor_buffer = flow.tensor_to_tensor_buffer(image_tensors, instance_dims=3)
        >>> flip_code = flow.ones(arr.shape[0], dtype=flow.int8)
        >>> output = nn.image.flip()(image_tensor_buffer, flip_code).numpy()
        >>> output[0]
        array([[[3., 2., 1.],
                [1., 2., 3.]],
        <BLANKLINE>
               [[4., 3., 2.],
                [2., 3., 4.]]], dtype=float32)
        >>> output[1]
        array([[[5., 4., 3.],
                [3., 4., 5.]],
        <BLANKLINE>
               [[6., 5., 4.],
                [4., 5., 6.]]], dtype=float32)
    """

    def __init__(self):
        super().__init__()

    def forward(self, images, flip_code):
        return flow._C.image_flip(images, flip_code=flip_code)


class ImageDecode(Module):
    def __init__(self, dtype: flow.dtype = flow.uint8, color_space: str = "BGR"):
        super().__init__()
        self.color_space = color_space
        self.dtype = dtype
        self._op = flow.stateful_op("image_decode").Input("in").Output("out").Build()

    def forward(self, input):
        return _C.dispatch_image_decode(
            self._op, input, color_space=self.color_space, data_type=self.dtype
        )


class ImageNormalize(Module):
    def __init__(self, std: Sequence[float], mean: Sequence[float]):
        super().__init__()
        self.std = std
        self.mean = mean
        self._op = flow.stateful_op("image_normalize").Input("in").Output("out").Build()

    def forward(self, input):
        return _C.dispatch_image_normalize(
            self._op, input, mean=self.mean, std=self.std
        )


class COCOReader(Module):
    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        batch_size: int,
        shuffle: bool = True,
        random_seed: Optional[int] = None,
        group_by_aspect_ratio: bool = True,
        remove_images_without_annotations: bool = True,
        stride_partition: bool = True,
        device: Union[flow.device, str] = None,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    ):
        super().__init__()
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.group_by_aspect_ratio = group_by_aspect_ratio
        self.remove_images_without_annotations = remove_images_without_annotations
        self.stride_partition = stride_partition
        if random_seed is None:
            random_seed = random.randrange(sys.maxsize)
        self.random_seed = random_seed

        self.placement = placement
        if placement is None:
            self.device = device or flow.device("cpu")
        else:
            if device is not None:
                raise ValueError(
                    "when param sbp is specified, param device should not be specified"
                )

            if isinstance(sbp, (tuple, list)):
                for sbp_item in sbp:
                    if not isinstance(sbp_item, flow.sbp.sbp):
                        raise ValueError(f"invalid sbp item: {sbp_item}")
            elif isinstance(sbp, flow.sbp.sbp):
                sbp = (sbp,)
            else:
                raise ValueError(f"invalid param sbp: {sbp}")

            if len(sbp) != len(placement.hierarchy):
                raise ValueError(
                    "dimensions of sbp and dimensions of hierarchy of placement don't equal"
                )
        self.sbp = sbp

        self._op = (
            flow.stateful_op("COCOReader")
            .Output("image")
            .Output("image_id")
            .Output("image_size")
            .Output("gt_bbox")
            .Output("gt_label")
            .Output("gt_segm")
            .Output("gt_segm_index")
            .Build()
        )

    def forward(self):
        if self.placement is None:
            # local apply
            outputs = _C.dispatch_coco_reader(
                self._op,
                session_id=flow.current_scope().session_id,
                annotation_file=self.annotation_file,
                image_dir=self.image_dir,
                batch_size=self.batch_size,
                shuffle_after_epoch=self.shuffle,
                random_seed=self.random_seed,
                group_by_ratio=self.group_by_aspect_ratio,
                remove_images_without_annotations=self.remove_images_without_annotations,
                stride_partition=self.stride_partition,
                device=self.device,
            )
        else:
            # consistent apply
            outputs = _C.dispatch_coco_reader(
                self._op,
                session_id=flow.current_scope().session_id,
                annotation_file=self.annotation_file,
                image_dir=self.image_dir,
                batch_size=self.batch_size,
                shuffle_after_epoch=self.shuffle,
                random_seed=self.random_seed,
                group_by_ratio=self.group_by_aspect_ratio,
                remove_images_without_annotations=self.remove_images_without_annotations,
                stride_partition=self.stride_partition,
                placement=self.placement,
                sbp=self.sbp,
            )
        return outputs


class ImageBatchAlign(Module):
    def __init__(self, shape: Sequence[int], dtype: flow.dtype, alignment: int):
        super().__init__()
        self._op = (
            flow.stateful_op("image_batch_align").Input("in").Output("out").Build()
        )
        self.shape = shape
        self.dtype = dtype
        self.alignment = alignment

    def forward(self, input):
        return _C.dispatch_image_batch_align(
            self._op,
            input,
            shape=self.shape,
            data_type=self.dtype,
            alignment=self.alignment,
            dynamic_out=False,
        )


class OFRecordBytesDecoder(Module):
    r"""This operator reads an tensor as bytes. The output might need

    further decoding process like cv2.imdecode() for images and decode("utf-8")

    for characters,depending on the downstream task.

    Args:
        blob_name: The name of the target feature in OFRecord.

        name: The name for this component in the graph.

        input: the Tensor which might be provided by an OFRecordReader.

    Returns:

        The result Tensor encoded with bytes.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> def example():
        ...      batch_size = 16
        ...      record_reader = flow.nn.OFRecordReader(
        ...         "dataset/",
        ...         batch_size=batch_size,
        ...         part_name_suffix_length=5,
        ...      )
        ...      val_record = record_reader()

        ...      bytesdecoder_img = flow.nn.OFRecordBytesDecoder("encoded")

        ...      image_bytes_batch = bytesdecoder_img(val_record)

        ...      image_bytes = image_bytes_batch.numpy()[0]
        ...      return image_bytes
        ... example()  # doctest: +SKIP
        array([255 216 255 ...  79 255 217], dtype=uint8)



    """

    def __init__(self, blob_name: str, name: Optional[str] = None):
        super().__init__()
        if name is not None:
            print("WARNING: name has been deprecated and has NO effect.\n")
        self._op = (
            flow.stateful_op("ofrecord_bytes_decoder").Input("in").Output("out").Build()
        )
        self.blob_name = blob_name

    def forward(self, input):
        return _C.dispatch_ofrecord_bytes_decoder(self._op, input, name=self.blob_name)


class GPTIndexedBinDataReader(Module):
    def __init__(
        self,
        data_file_prefix: str,
        seq_length: int,
        num_samples: int,
        batch_size: int,
        dtype: flow.dtype = flow.int64,
        shuffle: bool = True,
        random_seed: Optional[int] = None,
        split_sizes: Optional[Sequence[str]] = None,
        split_index: Optional[int] = None,
        device: Union[flow.device, str] = None,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    ):
        super().__init__()

        self.data_file_prefix = data_file_prefix
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.dtype = dtype
        self.shuffle = shuffle
        self.placement = placement
        if placement is None:
            self.device = device or flow.device("cpu")
        else:
            if device is not None:
                raise ValueError(
                    "when param sbp is specified, param device should not be specified"
                )

            if isinstance(sbp, (tuple, list)):
                for sbp_item in sbp:
                    if not isinstance(sbp_item, flow.sbp.sbp):
                        raise ValueError(f"invalid sbp item: {sbp_item}")
            elif isinstance(sbp, flow.sbp.sbp):
                sbp = (sbp,)
            else:
                raise ValueError(f"invalid param sbp: {sbp}")

            if len(sbp) != len(placement.hierarchy):
                raise ValueError(
                    "dimensions of sbp and dimensions of hierarchy of placement don't equal"
                )
        self.sbp = sbp

        if random_seed is None:
            random_seed = random.randrange(sys.maxsize)
        self.random_seed = random_seed

        if split_index is None:
            split_index = 0
        self.split_index = split_index

        if split_sizes is None:
            split_sizes = (1,)
        self.split_sizes = split_sizes

        if split_index >= len(split_sizes):
            raise ValueError(
                "split index {} is out of range, split_sizes {}".formart(
                    split_index, split_sizes
                )
            )

        self.op_ = (
            flow.stateful_op("megatron_gpt_mmap_data_loader").Output("out").Build()
        )

    def forward(self):
        if self.placement is None:
            output = _C.dispatch_megatron_gpt_mmap_data_loader(
                self.op_,
                data_file_prefix=self.data_file_prefix,
                seq_length=self.seq_length,
                label_length=1,
                num_samples=self.num_samples,
                batch_size=self.batch_size,
                dtype=self.dtype,
                shuffle=self.shuffle,
                random_seed=self.random_seed,
                split_sizes=self.split_sizes,
                split_index=self.split_index,
                device=self.device,
            )
        else:
            output = _C.dispatch_megatron_gpt_mmap_data_loader(
                self.op_,
                data_file_prefix=self.data_file_prefix,
                seq_length=self.seq_length,
                label_length=1,
                num_samples=self.num_samples,
                batch_size=self.batch_size,
                dtype=self.dtype,
                shuffle=self.shuffle,
                random_seed=self.random_seed,
                split_sizes=self.split_sizes,
                split_index=self.split_index,
                placement=self.placement,
                sbp=self.sbp,
            )
        return output


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
